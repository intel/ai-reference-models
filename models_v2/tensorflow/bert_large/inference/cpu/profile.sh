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

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, bfloat16 or fp16."
  exit 1
elif [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "bfloat16" ] && [ ${PRECISION} != "fp16" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, bfloat16 and fp16"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

# Unzip pretrained model files
if [[ -z "${CHECKPOINT_DIR}" ]]; then
  pretrained_model_dir="pretrained_model/bert_large_checkpoints"
  if [ ! -d "${pretrained_model_dir}" ]; then
    unzip pretrained_model/bert_large_checkpoints.zip -d pretrained_model
  fi
  CHECKPOINT_DIR="${MODEL_DIR}/${pretrained_model_dir}"
fi

PRETRAINED_MODEL=${PRETRAINED_MODEL-${MODEL_DIR}/pretrained_model/fp32_bert_squad.pb}

echo 'CHECKPOINT_DIR='$CHECKPOINT_DIR
echo 'PRETRAINED_MODEL='$PRETRAINED_MODEL

# Create an array of input directories that are expected and then verify that they exist
declare -A input_dirs
input_dirs[CHECKPOINT_DIR]=${CHECKPOINT_DIR}
input_dirs[DATASET_DIR]=${DATASET_DIR}

for i in "${!input_dirs[@]}"; do
  var_name=$i
  dir_path=${input_dirs[$i]}

  if [[ -z $dir_path ]]; then
    echo "The required environment variable $var_name is empty" >&2
    exit 1
  fi

  if [[ ! -d $dir_path ]]; then
    echo "The $var_name path '$dir_path' does not exist" >&2
    exit 1
  fi
done

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="32"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

source "${MODEL_DIR}/models_v2/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
    --model-name=bert_large \
    --precision=${PRECISION} \
    --mode=inference \
    --framework=tensorflow \
    --in-graph ${PRETRAINED_MODEL} \
    --batch-size=${BATCH_SIZE} \
    --data-location ${DATASET_DIR} \
    --checkpoint ${CHECKPOINT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    $@ \
    -- infer_option=SQuAD \
       profile=True
