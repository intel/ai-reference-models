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
# ============================================================================

CORES_PER_SOCKET = 56
NUM_SOCKETS = 2

FP16_WARMUP_IDX = 2348
FP16_WARMUP_LEN = 512
FP16_WARMUP_BS = 109
FP16_SATURATE_BS = 64

INT4_WARMUP_IDX = 2348
INT4_WARMUP_LEN = 512
INT4_WARMUP_BS = 105
INT4_SATURATE_BS = 64


class InputItem:
    def __init__(self, id, idx, input_ids=None, attn_masks=None, input_lens=None):
        self.id = id
        self.idx = idx
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.input_lens = input_lens


class OutputItem:
    def __init__(self, id, result, input_lens=None):
        self.id = id
        self.result = result
        self.input_lens = input_lens
