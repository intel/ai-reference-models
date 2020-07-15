#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
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

export PYTHONPATH=$MODEL_DIR/models/language_modeling/tensorflow/bert_large/training:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/mpirun:$LD_LIBRARY_PATH

echo 'BERT_BASE_DIR='$BERT_BASE_DIR
echo 'GLUE_DIR='$GLUE_DIR
echo 'MODEL_DIR='$MODEL_DIR
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'DATASET_DIR='$DATASET_DIR
echo 'PYTHONPATH='$PYTHONPATH
echo 'LD_LIBRARY_PATH='$LD_LIBRARY_PATH

_allow_run_as_root=''
if (( $(id -u) == 0 )); then
  _allow_run_as_root='--allow-run-as-root'
fi

if [[ -z $MODEL_DIR ]]; then
  echo "The required environment variable MODEL_DIR has not been set" >&2
  exit 1
fi

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set" >&2
  exit 1
fi

if [[ ! -d $OUTPUT_DIR ]]; then
  # Create the output directory, if it doesn't already exist
  mkdir -p $OUTPUT_DIR
fi

if [[ -z $DATASET_DIR ]]; then
  echo "The required environment variable DATASET_DIR has not been set" >&2
  exit 1
fi

if [[ ! -d $DATASET_DIR ]]; then
  echo 'The DATASET_DIR '$DATASET_DIR' does not exist' >&2
  exit 1
fi

mpirun -x PYTHONPATH -x LD_LIBRARY_PATH $_allow_run_as_root -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -np "2" -bind-to none -map-by slot \
  python3 $MODEL_DIR/models/language_modeling/tensorflow/bert_large/training/fp32/run_classifier.py \
    --task_name=MRPC \
    --do_train=true \
    --do_eval=true \
    --data_dir=$DATASET_DIR/$GLUE_DIR/MRPC \
    --vocab_file=$DATASET_DIR/$BERT_BASE_DIR/uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=$DATASET_DIR/$BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=$DATASET_DIR/$BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUTPUT_DIR \
    --use_tpu=False \
    --precision=fp32 \
    --do_lower_case=True
