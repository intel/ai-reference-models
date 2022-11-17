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

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

# Check for TF_MODELS_DIR
if [ -d "/tensorflow/models" ]; then # if true assume running in docker
   TF_MODELS_DIR=/tensorflow/models
elif [ -z ${TF_MODELS_DIR} ]; then
    echo "Please set TF_MODELS_DIR or run in docker mode." >&2
    exit 1
fi

echo 'TF_MODELS_DIR='$TF_MODELS_DIR
# Commenting out a line that causes a crash in the model script
sed -i.bak 's/atexit.register/# atexit.register/g' ${TF_MODELS_DIR}/official/recommendation/data_async_generation.py

if [ -z "${PRETRAINED_MODEL}" ]; then
  # Unzip pretrained model files
  pretrained_model_dir="pretrained_model/ncf_trained_movielens_1m"
  if [ ! -d "${pretrained_model_dir}" ]; then
    if [ -f pretrained_model/ncf_fp32_pretrained_model.tar.gz ]; then
      tar -C pretrained_model/ -xvf pretrained_model/ncf_fp32_pretrained_model.tar.gz
    else
      echo "The pretrained model could not be found. Please set the PRETRAINED_MODEL env var to point to the pretrained model directory."
      exit 1
    fi
  fi
  CHECKPOINT_DIR="${MODEL_DIR}/${pretrained_model_dir}"
else
  if [ ! -d "${PRETRAINED_MODEL}" ]; then
    echo "The PRETRAINED_MODEL directory (${PRETRAINED_MODEL}) does not exist."
    exit 1
  else
    CHECKPOINT_DIR=${PRETRAINED_MODEL}
  fi
fi

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
  BATCH_SIZE="256"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
      --checkpoint ${CHECKPOINT_DIR} \
      --data-location ${DATASET_DIR} \
      --model-source-dir ${TF_MODELS_DIR} \
      --model-name ncf \
      --socket-id 0 \
      --batch-size ${BATCH_SIZE} \
      --framework tensorflow \
      --precision fp32 \
      --mode inference \
      --output-dir ${OUTPUT_DIR} \
      $@
