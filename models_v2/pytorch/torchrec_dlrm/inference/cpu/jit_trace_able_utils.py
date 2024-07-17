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
from torchrec.models.dlrm import SparseArch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import List

def unpack(input :KeyedJaggedTensor) -> dict:
    output = {}
    for k, v in input.to_dict().items():
        output[k] = {}
        output[k]['values'] = v._values.int()
        output[k]['offsets'] = v._offsets.int()
    return output

class SparseArchTraceAbleWrapper(nn.Module):

    def __init__(self, sparse_arch: SparseArch) -> None:
        super().__init__()
        self.sparse_arch: SparseArch = sparse_arch

    def forward(
        self,
        features: dict,
    ) -> torch.Tensor:
        """
        Args:
            features (dict): input tensors of sparse features.

        Returns:
            torch.Tensor: tensor of shape B X F X D.
        """
        embedding_bag_collection = self.sparse_arch.embedding_bag_collection
        pooled_embeddings: List[torch.Tensor] = []
        for i, embedding_bag in enumerate(embedding_bag_collection.embedding_bags.values()):
            for feature_name in embedding_bag_collection._feature_names[i]:
                f = features[feature_name]
                res = embedding_bag(
                    f['values'],
                    f['offsets'],
                    per_sample_weights=None,
                ).float()
                pooled_embeddings.append(res)
        data = torch.cat(pooled_embeddings, dim=1)
        B: int = features['cat_0']['offsets'].numel() - 1
        return data.reshape(B, self.sparse_arch.F, self.sparse_arch.D)
