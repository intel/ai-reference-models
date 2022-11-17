#!/usr/bin/env bash
#
# Copyright (c) 2022 Intel Corporation
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
BATCH_SIZE=${BATCH_SIZE-1024}

echo 'MODEL_DIR='$MODEL_DIR
echo 'PRECISION='$PRECISION
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

if [[ ! -f "${FROZEN_GRAPH}" ]]; then
  pretrained_model=/workspace/tf-atsm-ssd-mobilenet-inference/pretrained_models/ssdmobilenet_${PRECISION}_pretrained_model_gpu.pb
else
  pretrained_model=${FROZEN_GRAPH}
fi

export TF_NUM_INTEROP_THREADS=1

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[PRECISION]=${PRECISION}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

WARMUP=""
if [[ $PRECISION == "int8" ]]; then
  WARMUP="-- warmup_steps=5 steps=20"
  else
  echo "ATS-M GPU SUPPORTS ONLY INT8 PRECISION"
  exit 1
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
    --in-graph ${pretrained_model} \
    --output-dir ${OUTPUT_DIR} \
    ${DATASET_OPTION} \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --precision ${PRECISION} \
    --mode inference \
    --benchmark-only \
    --batch-size=${BATCH_SIZE} \
    --gpu \
    $@ \
    ${WARMPUP} 
