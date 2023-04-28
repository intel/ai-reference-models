#!/bin/bash
#
# Copyright (c) 2022-2023 Intel Corporation
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

#https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/tree/main
#mkdir bert-large-uncased-whole-word-masking-finetuned-squad
set -e
mkdir squad_large_finetuned_checkpoint && cd squad_large_finetuned_checkpoint
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/pytorch_model.bin
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer_config.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt

