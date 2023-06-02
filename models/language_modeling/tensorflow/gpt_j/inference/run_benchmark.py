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

#

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import time

import datasets
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

import transformers
from transformers import (
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFAutoModelForCausalLM,
    TFTrainingArguments,
    set_seed,
)
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/tensorflow/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(TF_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to use.
    """

    model_name_or_path: Optional[str] = field(
        default="EleutherAI/gpt-j-6B",
        metadata={
            "help": (
                "The model checkpoint for GPT-J weights."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    precision: Optional[str] = field(
        default="fp32",
        metadata={"help": "The precision that we want to run with."},
    )
    batch_size: Optional[int] = field(
        default=100,
        metadata={"help": "Batch size for generation"},
    )
    max_output_tokens: Optional[int] = field(
        default=32,
        metadata={"help": "Maximum number of tokens to output"},
    )
    input_tokens: Optional[int] = field(
        default=32,
        metadata={"help": "Input tokens"},
    )
    warmup: Optional[float] = field(
        default=0.1,
        metadata={"help": "Warmup Steps"},
    )
    skip_rows: Optional[bool] = field(
        default=False,
        metadata={"help": "Skip some rows for latency use-case"},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for evaluation.
    """

    dataset_name: Optional[str] = field(
        default="EleutherAI/lambada_openai", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    model_args, data_args, run_args = parser.parse_args_into_dataclasses()
        
    if model_args.precision == "bfloat16":
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
    elif model_args.precision == "fp16":
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    
    if run_args.seed is not None:
        set_seed(run_args.seed)

    
    raw_datasets = load_dataset(
            data_args.dataset_name,
            cache_dir=model_args.checkpoint,
            use_auth_token=None,
        )
        
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    column_names = raw_datasets["test"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    mydata = tokenizer(raw_datasets["test"][text_column_name], padding='max_length', truncation=True, max_length=model_args.input_tokens, return_tensors="tf").input_ids
    if model_args.skip_rows:
        mydata = mydata[::100]
    else:
        mydata = mydata[::10]
    print(mydata.shape)
    
    model = TFAutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)
    model.compile()

    generate_kwargs = dict(do_sample=True, top_k=4, eos_token_id=model.config.eos_token_id)
    generate_kwargs["token_latency"] = True

    total_time = 0.0
    tmpval = mydata.shape[0] // model_args.batch_size
    num_iter =  tmpval if tmpval * model_args.batch_size == mydata.shape[0] else tmpval + 1
    
    num_warmup = (model_args.warmup * mydata.shape[0]) // model_args.batch_size
    num_warmup = num_warmup if num_warmup * model_args.batch_size == (model_args.warmup * mydata.shape[0]) else num_warmup + 1

    total_list = []

    for i in range(num_iter):
        tic = time.time()
        output = model.generate(
            mydata[i*model_args.batch_size: (i+1)*model_args.batch_size], max_new_tokens=32, **generate_kwargs
        )
        gen_ids = output[0]
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        toc = time.time()
        if i >= num_warmup:
            total_time += toc - tic
            total_list.append(output[1])
        print("{} / {} Done".format(min(mydata.shape[0], (i+1)*model_args.batch_size), mydata.shape[0]))

    

    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / (num_iter - num_warmup)

    first_latency = np.mean([x[0] for x in total_list])
    average_2n = list(chain(*[x[1:] for x in total_list]))
    average_2n.sort()
    average_2n_latency = np.mean(average_2n)

    if model_args.batch_size == 1:
        print("Inference latency: %.3f sec." % latency)
        print("First token average latency: %.3f sec." % first_latency)
        print("Rest tokens average latency: %.3f sec." % average_2n_latency)
    else:
        print("Average time spent per iteration (batch size = %i): %.3f sec." % (model_args.batch_size, latency))
        print("Average time spent to process the first token (batch size = %i): %.3f sec." % (model_args.batch_size, first_latency))
        print("Average time spent to process the rest tokens (batch size = %i): %.3f sec." % (model_args.batch_size, average_2n_latency))
    print("\n\n")
    tokens_processed = mydata.shape[1] * mydata.shape[0]

    throughput = tokens_processed / total_time
    print("Inference throughput (tokens / sec): %.3f" % throughput)

if __name__ == "__main__":
    main()
