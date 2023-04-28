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
echo 'PRECISION='$PRECISION
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'PRETRAINED_DIR='$PRETRAINED_DIR
echo 'SQUAD_DIR='$SQUAD_DIR
echo 'FROZEN_GRAPH='$FROZEN_GRAPH

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[PRECISION]=${PRECISION}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[PRETRAINED_DIR]=${PRETRAINED_DIR}
input_envs[SQUAD_DIR]=${SQUAD_DIR}
input_envs[FROZEN_GRAPH]=${FROZEN_GRAPH}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

# Check for precision
if [[ ${PRECISION} == "fp16" || ${PRECISION} == "fp32" ]]; then
  echo "The specified precision '${PRECISION}' is supported."
else
  echo "The specified precision '${PRECISION}' is not supported. Only fp16 and fp32 precision is supported"
  exit 1
fi

lspci_display_info=$(lspci | grep -i display)

if [ -z "${BATCH_SIZE}" ]; then
      BATCH_SIZE="64"
      echo "Running with default batch size of ${BATCH_SIZE}"
fi

export CreateMultipleSubDevices=1
export TF_NUM_INTEROP_THREADS=1

if [[ $PRECISION == "fp16" ]]; then
  export ITEX_AUTO_MIXED_PRECISION=1
  export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="FLOAT16"
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}
source "quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
    --model-name=bert_large \
    --precision=${PRECISION} \
    --mode=inference \
    --in-graph=${FROZEN_GRAPH} \
    --framework=tensorflow \
    --batch-size=${BATCH_SIZE} \
    --vocab-file=${PRETRAINED_DIR}/vocab.txt \
    --config-file=${PRETRAINED_DIR}/bert_config.json \
    --predict-file=${SQUAD_DIR}/dev-v1.1.json \
    --output-dir ${OUTPUT_DIR} \
    --accuracy-only \
    --gpu \
    $@ \
    -- infer_option=SQuAD
