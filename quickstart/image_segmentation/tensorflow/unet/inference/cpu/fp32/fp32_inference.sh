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

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

# Check for TF_UNET_DIR
if [ -d "/tf_unet" ]; then # if true assume running in docker
   TF_UNET_DIR=/tf_unet
elif [ -z ${TF_UNET_DIR} ]; then
    echo "Please set TF_UNET_DIR or run in docker mode." >&2
    exit 1
fi
echo 'TF_UNET_DIR='$TF_UNET_DIR

if [ -z "${PRETRAINED_MODEL}" ]; then
  # If the PRETRAINED_MODEL env var is not set, then we are probably running in a workload
  # container or model package, so check for the tar file and extract it, if needed
  pretrained_model_dir="pretrained_model/unet_trained"
  if [ ! -d "${pretrained_model_dir}" ]; then
    if [[ -f "pretrained_model/unet_fp32_pretrained_model.tar.gz" ]]; then
      mkdir -p pretrained_model
      tar -C pretrained_model/ -xvf pretrained_model/unet_fp32_pretrained_model.tar.gz
    else
      echo "The pretrained model was not found. Please set the PRETRAINED_MODEL var to point to the checkpoint file directory."
      exit 1
    fi
  fi
  PRETRAINED_MODEL="${MODEL_DIR}/${pretrained_model_dir}"
fi

# Create an array of input directories that are expected and then verify that they exist
declare -A input_dirs
input_dirs[PRETRAINED_MODEL]=${PRETRAINED_MODEL}

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
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
      --model-name unet \
      --precision fp32 \
      --mode inference \
      --framework tensorflow \
      --benchmark-only \
      --batch-size ${BATCH_SIZE} \
      --socket-id 0 \
      --checkpoint ${PRETRAINED_MODEL} \
      --model-source-dir ${TF_UNET_DIR} \
      --output-dir ${OUTPUT_DIR} \
      $@ \
      -- checkpoint_name=model.ckpt
