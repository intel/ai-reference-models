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

import sys
import os
import torch
import transformers
from transformers import AutoModelForCausalLM

# if len(sys.argv)<2:
#     print("Please provide a valid path for downloaded model")
#     print("usage : python download_gptj.py <path_where_to_save_model>")
#     exit()
# else:  
#     model_path = sys.argv[1]
#     if not os.path.exists(os.path.dirname(model_path)):
#         print("Error : Please provide a valid path")
#         exit()


model_path = os.environ.get('MODEL_DIR', "model")

os.makedirs(model_path, exist_ok=True)

model_name = "EleutherAI/gpt-j-6B"
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torchscript=True)  # torchscript will force `return_dict=False` to avoid jit errors
print("Loaded model")

model.save_pretrained(model_path)

print("Model downloaded and Saved in : ",model_path)
