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

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from utils import get_memory_usage, logger, jload


Buckets = dict()
MIN_CUTOFF = 64
MAX_CUTOFF = 1921
CUTOFF_STEP = 64
min_len = 1
for cutoff in range(MIN_CUTOFF, MAX_CUTOFF, CUTOFF_STEP):
    Buckets[cutoff] = list(range(min_len, cutoff, 1))
    min_len = cutoff

INPUT_BUCKETS = dict()
for cutoff, seq_lens in Buckets.items():
    for seq_len in seq_lens:
        INPUT_BUCKETS[seq_len] = cutoff

MAX_SAMPLES=1000 # maximum samples available in the dataset
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class Dataset(object):
    def __init__(self, model_path, dataset_path, total_count_override=-1, perf_count_override=-1, padding_side="left"):
        self.padding_side = padding_side
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side=self.padding_side,
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading dataset from {dataset_path} to Host")
        self.list_data_dict = jload(dataset_path)
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        self.sources = [prompt_input.format_map(example) for example in self.list_data_dict]
        self.targets = [f"{example['output']}" for example in self.list_data_dict]

        self.dataset = []
        for source in self.sources:
            input_sample = self.encode_sample(source)
            input_ids = input_sample.input_ids
            input_len = input_ids.shape[-1]
            attn_mask = torch.ones(input_len).view(1, input_len)
            self.dataset.append((input_ids, attn_mask, input_len))

        self.count = total_count_override if total_count_override > -1 else len(self.dataset)
        self.perf_count = perf_count_override if perf_count_override > -1 else self.count
        get_memory_usage("Host", "cpu")

    @torch.no_grad()
    def encode_sample(self, example):
        example = self.tokenizer([example], truncation=True, max_length=1919, return_tensors="pt", padding=True)
        return example

    def get_sample(self, query_idx_list):
        index = query_idx_list[0]
        input_ids, attn_mask, input_len = self.dataset[index]
        return (input_ids, attn_mask, input_len)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collect(self, batch_idxes, padded_batch_size=-1):
        actual_lens = [self.dataset[idx][2] for idx in batch_idxes]
        actual_batch_size = len(actual_lens)
        # pad T to max_len
        max_len = max(actual_lens)
        input_ids_list = []
        attn_mask_list = []
        for i, idx in enumerate(batch_idxes):
            pad_shape = (0, max_len-actual_lens[i]) if self.padding_side == "right" else (max_len-actual_lens[i], 0)
            input_ids_list.append(F.pad(self.dataset[idx][0], pad_shape, value=0))
            attn_mask_list.append(F.pad(self.dataset[idx][1], pad_shape, value=0))
        input_ids = torch.cat(input_ids_list)
        attn_masks = torch.cat(attn_mask_list)
        # pad N to padded_bs
        batch_size = max(actual_batch_size, padded_batch_size)
        input_ids = F.pad(input_ids, (0, 0, 0, batch_size-actual_batch_size), value=0)
        attn_masks = F.pad(attn_masks, (0, 0, 0, batch_size-actual_batch_size), value=0)
        input_lens = [max_len] * batch_size  # padding lens
        return input_ids, attn_masks, input_lens

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        pass
