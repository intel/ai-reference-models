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
import numpy as np
import datasets
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from collections import defaultdict

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



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for evaluation.
    """

    dataset_name: Optional[str] = field(
        default="EleutherAI/lambada_openai", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
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
            data_args.dataset_config_name,
            cache_dir=model_args.checkpoint,
            use_auth_token=None,
        )
        
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    column_names = raw_datasets["test"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    mydata = tokenizer(raw_datasets["test"][text_column_name], return_tensors="np").input_ids

    marg = {}
    stacked = np.concatenate(mydata)
    unique, counts = np.unique(stacked, return_counts=True)
    counts = counts / np.sum(counts)

    marg = dict(zip(unique, counts))
    marg = defaultdict(lambda: 0, marg)

    with run_args.strategy.scope():
        model = TFAutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, dtype=model_args.precision)
        embeddings = model.get_input_embeddings()

        if hasattr(embeddings, "embeddings"):
            embedding_size = embeddings.embeddings.shape[0]
        else:
            embedding_size = embeddings.weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        generate_kwargs = dict(do_sample=False, return_dict_in_generate=True, output_scores=True, eos_token_id=model.config.eos_token_id)
        
        model.compile(jit_compile=run_args.xla)
        start_time = time.time()
        tot = 0
        sc = 0
        for x in mydata:
            gen = model.generate([x[:-1]], max_new_tokens = 1, **generate_kwargs)
            dd = tf.nn.log_softmax(gen["scores"][0][0])
            sc += dd[x[-1]].numpy()
            if x[-1] == gen.sequences[0][-1].numpy():
                tot+=1
        sc = -1*sc
        sc = sc / mydata.shape[0]
        acc = tot / mydata.shape[0]

        print("Time taken {}".format(time.time() - start_time))
        print("Loss = {}".format(sc))
        print("Accuracy = {}".format(acc))
        print("Perplexity = {}".format(math.exp(sc)))
    

if __name__ == "__main__":
    main()
