#!/usr/bin/env bash
#
# Copyright (c) 2023 Intel Corporation
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
echo 'PRECISION='$PRECISION
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

export TF_NUM_INTEROP_THREADS=1
export ITEX_LAYOUT_OPT=1

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[MODEL_NAME]=${MODEL_NAME}
input_envs[IMAGE_FILE]=${IMAGE_FILE}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

mkdir -p ${OUTPUT_DIR}

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="64"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

echo "${MODEL_NAME} FP16 inference"

python -u $MODEL_DIR/models/image_recognition/tensorflow/efficientnet/inference/gpu/predict.py -m ${MODEL_NAME} -b ${BATCH_SIZE} -i ${IMAGE_FILE}
