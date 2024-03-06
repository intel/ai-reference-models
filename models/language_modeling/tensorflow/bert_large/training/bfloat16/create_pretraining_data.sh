
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

date
if [ -z "$BERT_BASE_DIR" ]; then 
  echo "ERROR: empty BERT_BASE_DIR" 
fi
if [ -z "$GLUE_DIR" ]; then 
  echo "ERROR: empty GLUE_DIR" 
fi
export TF_CPP_MIN_VLOG_LEVEL=0
export MKL_DNN_VERBOSE=0
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=./output/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=10

date
