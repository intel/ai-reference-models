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
from typing import Optional, Literal
import time

import datasets
import tensorflow as tf
from datasets import load_dataset
import numpy as np

import transformers
from transformers import (
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFAutoModelForCausalLM,
)
from transformers.utils.versions import require_version
from tensorflow.python.platform import tf_logging

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r benchmarks/language_modeling/tensorflow/gpt_j/requirements.txt")
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
    input_tokens: Literal[32, 64, 128, 256, 512, 1024, 2016] = 32
    warmup_steps: Optional[int] = field(
        default=10,
        metadata={"help": "Warmup Steps"},
    )
    dummy_data: Optional[bool] = field(
        default=True,
        metadata={"help": "Use dummy data for benchmarking"},
    )
    steps: Optional[int] = field(
        default=100,
        metadata={"help": "Steps to run the benchmarking for"},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory that stores the output logs"},
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
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
        
    if model_args.precision == "bfloat16":
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
    elif model_args.precision == "fp16":
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    nrows_warmup = model_args.warmup_steps * model_args.batch_size
    nrows_actual = model_args.steps * model_args.batch_size 
    nrows = nrows_warmup + nrows_actual
    
    if model_args.dummy_data:
        rdata = ["Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun." * (model_args.input_tokens // 32)] 
        rdata = rdata * nrows
        mydata = tokenizer(rdata, return_tensors="tf").input_ids
    else:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            cache_dir=model_args.checkpoint,
            use_auth_token=None,
        )
        column_names = raw_datasets["test"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        rdata = raw_datasets["test"][text_column_name]
        if len(rdata) < nrows:
            rdata = rdata * math.ceil(nrows / len(rdata))
        rdata = rdata[:nrows]
        mydata = tokenizer(rdata, padding='max_length', truncation=True, max_length=model_args.input_tokens, return_tensors="tf").input_ids
    
    
    print(mydata.shape)

    model = TFAutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, dtype=model_args.precision)
    model.compile()

    generate_kwargs = dict(do_sample=False, num_beams=4, eos_token_id=model.config.eos_token_id)
    generate_kwargs["token_latency"] = True
    
    total_time = 0.0
    
    total_list = []
    tgen = 0
    wgen = 0
    fgen = 0

    gen = tf.function(lambda x: model.generate(x, max_new_tokens=model_args.max_output_tokens, **generate_kwargs))

    for i in range(model_args.warmup_steps):
        tf_logging.warn('---> Start Warmup iteration {0}'.format(str(i+1)))
        tic = time.time()
        output = gen(mydata[i*model_args.batch_size: (i+1)*model_args.batch_size])
        toc = time.time()
        print(toc - tic)
        print(np.diff(output[1]), np.sum(np.diff(output[1])))
        decoded = tokenizer.batch_decode(output[0], skip_special_tokens=True)
        print(decoded)
        tf_logging.warn('---> Stop Warmup iteration {0}'.format(str(i+1)))
        
    print("\n\n")

    j = 0
    for i in range(model_args.warmup_steps, model_args.steps + model_args.warmup_steps):
        tf_logging.warn('---> Start iteration {0}'.format(str(j+1)))
        tic = time.time()
        output = gen(mydata[i*model_args.batch_size: (i+1)*model_args.batch_size])
        toc = time.time()
        print(toc - tic)
        print(np.diff(output[1]), np.sum(np.diff(output[1])))
        decoded = tokenizer.batch_decode(output[0], skip_special_tokens=True)
        print(decoded)
        tf_logging.warn('---> Stop iteration {0}'.format(str(j+1)))
        gen_ids = output[0]
        total_time += toc - tic
        wgen += (gen_ids.shape[1] - model_args.input_tokens) * gen_ids.shape[0]
        tgen += (gen_ids.shape[1] - model_args.input_tokens - 1) * gen_ids.shape[0]
        fgen += gen_ids.shape[0]
        total_list.append(np.diff(output[1]))
        print("{} / {} Done".format((j+1)*model_args.batch_size, mydata[nrows_warmup:].shape[0]))
        j+=1

    print("\n", "-" * 10, "Summary:", "-" * 10)
    tpi = total_time / (model_args.steps)

    first_latency = np.mean([x[0] for x in total_list])
    first_total = np.sum([x[0] for x in total_list])
    average_2n = list(chain(*[x[1:] for x in total_list]))
    average_2n.sort()

    average_2n_batch = np.mean(average_2n)
    rt = np.sum(average_2n)
    average_2n_latency = rt / tgen

    if model_args.batch_size == 1:
        print("Inference latency: %.5f sec." % tpi)
        print("First token average latency: %.5f sec." % first_latency)
        print("Rest tokens average latency: %.5f sec." % average_2n_latency)
    else:
        print("Average time spent per iteration (batch size = %i): %.5f sec." % (model_args.batch_size, tpi))
        print("Average time spent to process the first token (batch size = %i): %.5f sec." % (model_args.batch_size, first_latency))
        print("Average time spent to process 2 to rest tokens together (batch size = %i): %.5f sec." % (model_args.batch_size, average_2n_batch))
    print("\n\n")
    
    throughput = tgen / rt
    tpwhole = wgen / total_time
    tpfirst = fgen / first_total
    print("Inference generation throughput (first token) (tokens / sec): %.3f" % tpfirst)
    print("Inference generation throughput (2 to rest) (tokens / sec): %.3f" % throughput)
    print("\n")
    print("Inference generation throughput (tokens / sec): %.3f" % tpwhole)

if __name__ == "__main__":
    main()
