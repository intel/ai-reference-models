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

MODEL_DIR=${MODEL_DIR-$PWD}

echo 'MODEL_DIR='$MODEL_DIR
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'DATASET_DIR='$DATASET_DIR
echo 'PYTHONPATH='$PYTHONPATH
echo 'LD_LIBRARY_PATH='$LD_LIBRARY_PATH

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

BERT_BASE_DIR=$DATASET_DIR/dataset/bert_official/MRPC
GLUE_DIR=$DATASET_DIR/dataset/bert_official

python benchmarks/launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=32 \
    -- train-option=Classifier \
       task-name=MRPC \
       do-train=true \
       do-eval=true \
       data-dir=$GLUE_DIR/MRPC \
       vocab-file=$BERT_BASE_DIR/vocab.txt \
       config-file=$BERT_BASE_DIR/bert_config.json \
       init-checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
       max-seq-length=128 \
       learning-rate=2e-5 \
       num-train-epochs=30 \
       output-dir=/tmp/mrpc_output/ \
       optimized_softmax=True \
       experimental_gelu=True \
       do-lower-case=True
