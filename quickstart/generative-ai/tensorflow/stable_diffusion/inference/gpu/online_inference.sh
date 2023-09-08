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
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'PRECISION='$PRECISION

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs

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

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi
echo "Stable Diffusion Inference Inference"
if [[ ${PRECISION} == "fp32" || ${PRECISION} == "fp16" ]]; then
  python -u ${MODEL_DIR}/models/generative-ai/tensorflow/stable_diffusion/inference/gpu/stable_diffusion_inference.py --precision ${PRECISION} --store_result_dir ${OUTPUT_DIR} 2>&1 | tee $OUTPUT_DIR/${PRECISION}_stable_diffusion_logs.txt
else
  echo "Stable Diffusion currently supports fp32 and fp16 precisions."
  exit 1
fi
