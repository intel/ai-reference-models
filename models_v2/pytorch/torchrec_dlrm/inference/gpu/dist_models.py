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
import argparse
from functools import partial
import torch
from torchrec import EmbeddingBagCollection
import extend_distributed as ext_dist
from torchrec.models.dlrm import DLRM, InteractionDCNArch, OverArch
from typing import Dict, List, Optional, Tuple
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor
from torchrec.modules.crossnet import LowRankCrossNet
import time


def dist_embedding_forward(
    self,
    dense_features: torch.Tensor,
    sparse_features: KeyedJaggedTensor,
) -> torch.Tensor:
    """
    Args:
        dense_features (torch.Tensor): the dense features.
        sparse_features (KeyedJaggedTensor): the sparse features.

    Returns:
        torch.Tensor: logits.
    """
    indices = tuple(
        [
            sf.values().to(self.input_device)
            for _, sf in sparse_features.to_dict().items()
        ]
    )
    offsets = tuple(
        [
            sf.offsets().to(self.input_device)
            for _, sf in sparse_features.to_dict().items()
        ]
    )
    embedded_sparse = self.sparse_arch(indices, offsets)
    embedded_dense = self.dense_arch(dense_features)

    if self.fp16:
        embedded_sparse = (
            embedded_sparse.half()
            if embedded_sparse.dtype != torch.float16
            else embedded_sparse
        )
        embedded_dense = (
            embedded_dense.half()
            if embedded_dense.dtype != torch.float16
            else embedded_dense
        )
    elif self.bf16:
        embedded_sparse = (
            embedded_sparse.bfloat16()
            if embedded_sparse.dtype != torch.bfloat16
            else embedded_sparse
        )
        embedded_dense = (
            embedded_dense.bfloat16()
            if embedded_dense.dtype != torch.bfloat16
            else embedded_dense
        )

    concatenated_dense = self.inter_arch(
        dense_features=embedded_dense, sparse_features=embedded_sparse
    )
    logits = self.over_arch(concatenated_dense)
    return logits


def replace_embeddingbag_collection(model, device, args):
    from intel_extension_for_pytorch.nn import DistMergeEmbeddingBagWithAdaGrad

    optimizer_param = model.parameters()
    if args.ipex_optimize:
        new_collection = None
        new_forward_func = None
        params = []
        collection = list(
            model.sparse_arch.embedding_bag_collection.embedding_bags.values()
        )
        if not args.inference_only:
            if args.ipex_dist_merged_emb_adagrad:
                new_collection = (
                    DistMergeEmbeddingBagWithAdaGrad.from_embeddingbag_list(
                        collection, device=device, lr=args.learning_rate, eps=args.eps
                    )
                )
                new_forward_func = dist_embedding_forward
            else:
                pass

        if new_collection:
            model.sparse_arch = new_collection
            if args.ipex_dist_merged_emb_adagrad:
                setattr(model, "forward", partial(new_forward_func, model))
                setattr(model, "input_device", device)
                setattr(model, "bf16", args.bf16)
                setattr(model, "fp16", args.fp16)
                for name, param in model.named_parameters():
                    if name != "sparse_arch.weights.0":
                        params.append(param)
                optimizer_param = params
            del collection
    return model, optimizer_param


class DIST_DLRM_DCN(DLRM):
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

    Example::

        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2",
           embedding_dim=D,
           num_embeddings=100,
           feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = DLRM_DCN(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20, D],
           dcn_num_layers=2,
           dcn_low_rank_dim=8,
           over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f3"],
           values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
           offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
           dense_features=features,
           sparse_features=sparse_features,
        )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        num_total_sparse_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dcn_num_layers: int,
        dcn_low_rank_dim: int,
        n_emb: int,
        shard_matrix,
        shard_strategy,
        args: argparse.Namespace,
        dense_device: Optional[torch.device] = None,
    ) -> None:
        # initialize DLRM
        # sparse arch and dense arch are initialized via DLRM
        super().__init__(
            embedding_bag_collection,
            dense_in_features,
            dense_arch_layer_sizes,
            over_arch_layer_sizes,
            dense_device,
        )

        self.args = args
        if ext_dist.my_size > 1:
            if n_emb < ext_dist.my_size:
                sys.exit(
                    "only (%d) sparse features for (%d) devices, table partitions will fail"
                    % (n_emb, ext_dist.my_size)
                )
            self.n_global_emb = n_emb
            # self.n_local_emb, self.n_emb_per_rank = ext_dist.get_split_lengths(
            #    n_emb
            # )
            self.shard_matrix = shard_matrix
            self.shard_strategy = shard_strategy
            self.n_emb_per_rank = [len(s_m) for s_m in shard_matrix]
            # self.local_emb_slice = ext_dist.get_my_slice(n_emb)
            # self.local_emb_indices = list(range(n_emb))[self.local_emb_slice]

        self.embedding_dim: int = embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim
        # num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        # Fix interaction and over arch for DLRM_DCN

        crossnet = LowRankCrossNet(
            in_features=(num_total_sparse_features + 1) * self.embedding_dim,
            num_layers=dcn_num_layers,
            low_rank=dcn_low_rank_dim,
        )

        self.inter_arch = InteractionDCNArch(
            num_sparse_features=num_total_sparse_features,
            crossnet=crossnet,
        )
        self.inter_arch.to(dense_device)

        over_in_features: int = (num_total_sparse_features + 1) * self.embedding_dim

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )

    def sequential_forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits.
        """
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features)
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        return logits

    def distributed_forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits.
        """
        embedded_sparse = self.sparse_arch(sparse_features)
        a2a_req = ext_dist.alltoall(embedded_sparse, self.n_emb_per_rank)
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = a2a_req.wait()
        # ['0', '8', '16', '24'], ['1', '9', '17', '25'], ['2', '10', '18'], ['3', '11', '19'], ['4', '12', '20'], ['5', '13', '21'], ['6', '14', '22'], ['7', '15', '23']
        split_embed = {}
        for index_rank in range(len(self.shard_matrix)):
            embed_rank = torch.split(
                embedded_sparse[index_rank], self.embedding_dim, dim=1
            )
            for index in range(len(self.shard_matrix[index_rank])):
                split_embed[self.shard_matrix[index_rank][index]] = embed_rank[index]

        embedded_sparse = []
        for i in range(len(self.args.num_embeddings_per_feature)):
            embedded_sparse.append(split_embed[i])
        embedded_sparse = torch.cat(list(embedded_sparse), dim=1).reshape(
            embedded_dense.size(0), -1, embedded_dense.size(-1)
        )
        if self.args.fp16:
            embedded_sparse = (
                embedded_sparse.half()
                if embedded_sparse.dtype != torch.float16
                else embedded_sparse
            )
            embedded_dense = (
                embedded_dense.half()
                if embedded_dense.dtype != torch.float16
                else embedded_dense
            )
        elif self.args.bf16:
            embedded_sparse = (
                embedded_sparse.bfloat16()
                if embedded_sparse.dtype != torch.bfloat16
                else embedded_sparse
            )
            embedded_dense = (
                embedded_dense.bfloat16()
                if embedded_dense.dtype != torch.bfloat16
                else embedded_dense
            )

        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        return logits

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        if ext_dist.my_size > 1:
            # multi-node multi-device run
            return self.distributed_forward(dense_features, sparse_features)
        else:
            # single device run
            return self.sequential_forward(dense_features, sparse_features)
