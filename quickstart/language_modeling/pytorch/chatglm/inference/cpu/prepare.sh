#!/usr/bin/env bash
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

if [ -z "$TRANSFORMERS_CACHE" ]; then
    TRANSFORMERS_CACHE=~/.cache/huggingface/hub/
else
    echo "TRANSFORMERS_CACHE is set to: $TRANSFORMERS_CACHE"
fi

huggingface-cli download "THUDM/chatglm3-6b" "config.json" "tokenizer_config.json"
directory=${TRANSFORMERS_CACHE}/models--THUDM--chatglm3-6b/snapshots/

latest_dir=$(ls -td ${directory}/*/ | head -n1)
# modify config.json
sed -i "s/\"torch_dtype\":\ \"float16\"/\"torch_dtype\":\ \"float32\"/g" "${latest_dir}/config.json"
# modify tokenizer_config.json
sed -i "s/\"THUDM\/chatglm3-6b--tokenization_chatglm.ChatGLMTokenizer\"/\"tokenization_chatglm.ChatGLMTokenizer\"/g" "${latest_dir}/tokenizer_config.json"
new_line='"name_or_path": "THUDM/chatglm3-6b",'
if ! grep -q "${new_line}" "${latest_dir}/tokenizer_config.json"; then
    sed -i '1a\'$'\n'''"$new_line"'' "${latest_dir}/tokenizer_config.json"
fi
