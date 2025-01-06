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
import time
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging
from pathlib import Path

import transformers

import intel_extension_for_pytorch as ipex
from typing import Optional, Tuple, Union


torch._C._jit_set_texpr_fuser_enabled(False)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BACKEND")

class Backend(object):

    def __init__(self, model_checkpoint=None, precision="int8", quantized_model=None):

        self.model_checkpoint = model_checkpoint
        self.precision = precision
        self.generate_kwargs = {
            "early_stopping": True,
            "max_new_tokens": 128,
            "min_new_tokens": 30,
            "num_beams": 4,
        }
        self.amp_enabled = False
        self.amp_dtype = None

        self.quantized_model = quantized_model

        if self.precision in ["int8", "int4_bf16_mixed"]:
            if self.quantized_model is None:
                print("Running int8 requires a 'quantized_model' path")
                sys.exit(1)
            elif not os.path.isfile(self.quantized_model):
                print("Path to quantized model {} not found".format(self.quantized_model))
                sys.exit(1)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,
            )
        
    def loadModel(self):
        """ Loads the pretrained model """

        config = AutoConfig.from_pretrained(self.model_checkpoint, torchscript=True)
        with ipex.OnDevice(dtype=torch.bfloat16, device="meta"):
            self.model = AutoModelForCausalLM.from_config(config)

        self.model = self.model.to(memory_format=torch.channels_last)
        self.model = self.model.eval()

        qconfig = ipex.quantization.default_static_qconfig_mapping
        self.model = ipex.optimize_transformers(
            self.model.eval(),
            dtype=torch.float,
            inplace=True,
            quantization_config=qconfig,
            deployment_mode=False
        )
        torch._C._jit_set_texpr_fuser_enabled(False)
        self.quant_model = torch.jit.load(self.quantized_model)
        self.quant_model = torch.jit.freeze(self.quant_model.eval())
        ipex._set_optimized_model_for_generation(self.model, optimized_model=self.quant_model)

    def predict(self, input_batch, attention_mask=None):
        """ Runs inference on 'input_batch' """
        enable_autocast = self.precision in ["bf16", "mix", "int4_bf16_mixed"]
        with torch.inference_mode(), torch.no_grad(), torch.autocast("cpu", enabled=enable_autocast, dtype=torch.bfloat16):
            outputs = self.model.generate(input_batch, attention_mask=attention_mask,
                    **self.generate_kwargs,
                    pad_token_id=self.tokenizer.pad_token_id
                    )
        return outputs


