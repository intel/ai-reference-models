#!/usr/bin/env bash
#
# Copyright (c) 2021 Intel Corporation
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

set -e

echo "----------- Start: DUMMY DATA generation --------------"
# Assumption is BERT_LARGE_DIR, DUMMY_DATA and MODEL_DIR is set.
SOURCE_DIR=${MODEL_DIR}/models/language_modeling/tensorflow/bert_large/training/fp32

pushd $SOURCE_DIR

python create_pretraining_data.py \
        --input_file=./sample_text.txt \
        --output_file=$DUMMY_DATA \
        --vocab_file=$BERT_LARGE_DIR/vocab.txt \
        --do_lower_case=False \
        --max_seq_length=512 \
        --max_predictions_per_seq=76 \
        --masked_lm_prob=0.15 \
        --random_seed=12345 \
        --dupe_factor=5

popd

echo "----------- End: DUMMY DATA generation --------------"
