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

import math
import torch
from torch.nn.functional import pad

# Dictionary of Hugging Face models
hf_model_map = {
    "distilgpt2": {
        "lora_target_modules": [],
    },
    "EleutherAI/gpt-j-6b": {
        "lora_target_modules": ["q_proj", "v_proj"],
    },
    "bigscience/bloom-560m": {
        "lora_target_modules": ["query_key_value"],
    },
    "bigscience/bloomz-3b": {
        "lora_target_modules": ["query_key_value"],
    },
    "bigscience/bloomz-560m": {
        "lora_target_modules": ["query_key_value"],
    },
}


class INCDataloader:
    def __init__(
        self,
        dataset,
        tokenizer,
        batch_size=1,
        device="cpu",
        max_seq_length=512,
        for_calib=False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.for_calib = for_calib
        self.length = math.ceil(len(dataset) / self.batch_size)
        self.pad_len = 196

        self.dataset.set_format(type="torch", columns=["input_ids"])

    def pad_input(self, input):
        input_id = input["input_ids"].unsqueeze(0)
        label = input_id[:, -1].to(self.device)
        pad_len = self.pad_len - input_id.shape[1]
        label_index = -2 - pad_len
        input_id = pad(input_id, (0, pad_len), value=1)

        return (input_id, label, label_index)

    def __iter__(self):
        input_ids = None
        labels = None
        label_indices = None
        for idx, batch in enumerate(self.dataset):
            input_id, label, label_index = self.pad_input(batch)

            if input_ids is None:
                input_ids = input_id
                labels = label
                label_indices = [label_index]
            else:
                input_ids = torch.cat((input_ids, input_id), 0)
                labels = torch.cat((labels, label), 0)
                label_indices.append(label_index)

            if (idx + 1) % self.batch_size == 0:
                if self.for_calib:
                    if input_ids.shape[1] > self.max_seq_length:
                        input_ids = input_ids[:, self.max_seq_length]
                    yield input_ids
                else:
                    yield (input_ids, labels, label_indices)
                input_ids = None
                labels = None
                label_indices = None
        if (idx + 1) % self.batch_size != 0:
            if self.for_calib:
                if input_ids.shape[1] > self.max_seq_length:
                    input_ids = input_ids[:, self.max_seq_length]
                yield input_ids
            else:
                yield (input_ids, labels, label_indices)

    def __len__(self):
        return self.length
