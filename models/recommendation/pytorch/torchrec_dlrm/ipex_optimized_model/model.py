#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from torch import nn

from torchrec.models.dlrm import SparseArch, InteractionDCNArch, DLRM_DCN
from torchrec.modules.embedding_modules import EmbeddingBagCollection

from typing import Dict, List, Optional, Tuple
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.nn.modules import MergedEmbeddingBagWithCat, MergedEmbeddingBagWithAdaGrad, DistMergeEmbeddingBagWithAdaGrad

class _LowRankCrossNet(torch.nn.Module):
    def __init__(
        self,
        lr_crossnet,
    ) -> None:
        super().__init__()
        self._num_layers = lr_crossnet._num_layers
        self._in_features = lr_crossnet.bias[0].shape[0]
        self._low_rank = lr_crossnet._low_rank
        self.V_linears = torch.nn.ModuleList()
        self.W_linears = torch.nn.ModuleList()
        for i in range(self._num_layers):
            self.V_linears.append(torch.nn.Linear(self._in_features, self._low_rank, bias=False))
            self.W_linears.append(torch.nn.Linear(self._low_rank, self._in_features, bias=True))
            self.V_linears[i].weight.data = lr_crossnet.V_kernels[i]
            self.W_linears[i].weight.data = lr_crossnet.W_kernels[i]
            self.W_linears[i].bias.data = lr_crossnet.bias[i]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_0 = input
        x_l = x_0
        for layer in range(self._num_layers):
            x_l_v = self.V_linears[layer](x_l)
            x_l_w = self.W_linears[layer](x_l_v)
            x_l = x_0 * x_l_w + x_l  # (B, N)
        return x_l

def replace_crossnet(dlrm):
    crossnet = dlrm.inter_arch.crossnet
    new_crossnet = _LowRankCrossNet(crossnet)
    dlrm.inter_arch.crossnet = new_crossnet
    del(crossnet)

class SparseArchCatDense(SparseArch):

    def forward(
        self,
        embedded_dense_features,
        sparse_features,
    ) -> torch.Tensor:
        """
        Args:
            embedded_dense_features: the output of DenseArch.
            sparse_features: the indices/offsets for F embeddingbags in embedding_bag_collection

        Returns:
            torch.Tensor: tensor of shape B X (F + 1) X D.
        """
        (B, _) = embedded_dense_features.shape
        embedding_bag_collection = self.embedding_bag_collection
        indices = tuple([sf['values'] for _, sf in sparse_features.items()])
        offsets = tuple([sf['offsets'] for _, sf in sparse_features.items()])
        if isinstance(embedding_bag_collection, MergedEmbeddingBagWithCat):
            return embedding_bag_collection(indices, offsets, embedded_dense_features)
        elif isinstance(embedding_bag_collection, DistMergeEmbeddingBagWithAdaGrad):
            embedded_sparse_features = embedding_bag_collection(indices, offsets)
            return torch.cat((embedded_dense_features.unsqueeze(1), embedded_sparse_features), dim=1).reshape(B, -1)
        elif isinstance(embedding_bag_collection, MergedEmbeddingBagWithAdaGrad):
            embedded_sparse_features = embedding_bag_collection(indices, offsets)
        else:
            embedded_sparse_features: List[torch.Tensor] = []
            for i, embedding_bag in enumerate(embedding_bag_collection.embedding_bags.values()):
                for feature_name in embedding_bag_collection._feature_names[i]:
                    f = sparse_features[feature_name]
                    res = embedding_bag(
                        f['values'],
                        f['offsets'],
                        per_sample_weights=None,
                    )
                    embedded_sparse_features.append(res)
        to_cat = [embedded_dense_features] + list(embedded_sparse_features)
        out = torch.cat(to_cat, dim=1)
        return out

def replace_embeddingbag_collection(dlrm, args):
    if args.ipex_optimize:
        new_collection = None
        collection = list(dlrm.sparse_arch.embedding_bag_collection.embedding_bags.values())
        if args.inference_only and args.ipex_merged_emb_cat:
            new_collection = MergedEmbeddingBagWithCat.from_embeddingbag_list(collection)
        if not args.inference_only:
            if args.ipex_dist_merged_emb_adagrad:
                new_collection = DistMergeEmbeddingBagWithAdaGrad.from_embeddingbag_list(collection, lr=args.learning_rate, eps=1e-8)
            elif args.ipex_merged_emb_adagrad:
                new_collection = MergedEmbeddingBagWithAdaGrad.from_embeddingbag_list(collection, lr=args.learning_rate, eps=1e-8)
            else:
                for emb in collection:
                    emb.sparse = True

        if new_collection:
            dlrm.sparse_arch.embedding_bag_collection = new_collection
            del(collection)

class InteractionDCNArchWithoutCat(InteractionDCNArch):

    def forward(
        self, concat_dense_sparse: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            concat_dense_sparse (torch.Tensor): an input tensor of size B X (F*D + D).

        Returns:
            torch.Tensor: an output tensor of size B X (F*D + D).
        """

        # size B X (F * D + D)
        return self.crossnet(concat_dense_sparse)

class IPEX_DLRM_DCN(DLRM_DCN):
    """
    Recsys model with DCN modified from the original model from "Deep Learning Recommendation
    Model for Personalization and Recommendation Systems"
    (https://arxiv.org/abs/1906.00091). Similar to DLRM module but has
    DeepCrossNet https://arxiv.org/pdf/2008.13535.pdf as the interaction layer.

    The module assumes all sparse features have the same embedding dimension
    (i.e. each EmbeddingBagConfig uses the same embedding_dim).

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the `DenseArch`.
        over_arch_layer_sizes (List[int]): the layer sizes for the `OverArch`.
            The output dimension of the `InteractionArch` should not be manually
            specified here.
        dcn_num_layers (int): the number of DCN layers in the interaction.
        dcn_low_rank_dim (int): the dimensionality of low rank approximation
            used in the dcn layers.
        dense_device (Optional[torch.device]): default compute device.
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dcn_num_layers: int,
        dcn_low_rank_dim: int,
        dense_device: Optional[torch.device] = None,
    ) -> None:
        # initialize DLRM
        # sparse arch and dense arch are initialized via DLRM
        super().__init__(
            embedding_bag_collection,
            dense_in_features,
            dense_arch_layer_sizes,
            over_arch_layer_sizes,
            dcn_num_layers,
            dcn_low_rank_dim,
            dense_device,
        )

        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        embedding_bag_collection = self.sparse_arch.embedding_bag_collection
        self.sparse_arch = SparseArchCatDense(embedding_bag_collection)

        crossnet = self.inter_arch.crossnet
        self.inter_arch = InteractionDCNArchWithoutCat(
            num_sparse_features=num_sparse_features,
            crossnet=crossnet,
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features,
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits.
        """
        embedded_dense = self.dense_arch(dense_features)
        concat_sparse_dense = self.sparse_arch(embedded_dense, sparse_features)
        concatenated_dense = self.inter_arch(concat_sparse_dense)
        logits = self.over_arch(concatenated_dense)
        return logits
