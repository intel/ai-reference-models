#!/bin/bash

#
# Copyright (c) 2024 Intel Corporation
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

# Clone the Transformers repo in the gptj 6b inference directory
cd ${MODEL_DIR}
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.38.1
git apply ${MODEL_DIR}/../../../../common/enable_ipex_for_transformers.diff
pip install -e ./
cd ..

# Get prompt.json for gneration inference
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json
mv prompt.json ${MODEL_DIR}
