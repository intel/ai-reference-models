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

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

# Use synthetic data (no --data-location arg) if no DATASET_DIR is set
dataset_arg="--data-location=${DATASET_DIR}"
if [ -z "${DATASET_DIR}" ]; then
  echo "Using synthetic data, since the DATASET_DIR environment variable is not set."
  dataset_arg=""
elif [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${PRETRAINED_MODEL}" ]; then
  PRETRAINED_MODEL="${MODEL_DIR}/mobilenetv1_int8_pretrained_model.pb"

  if [[ ! -f "${PRETRAINED_MODEL}" ]]; then
    echo "The pretrained model could not be found. Please set the PRETRAINED_MODEL env var to point to the frozen graph file."
    exit 1
  fi
elif [[ ! -f "${PRETRAINED_MODEL}" ]]; then
  echo "The file specified by the PRETRAINED_MODEL environment variable (${PRETRAINED_MODEL}) does not exist."
  exit 1
fi

CORES_PER_INSTANCE="4"

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name mobilenet_v1 \
  --precision int8 \
  --mode inference \
  --framework tensorflow \
  --benchmark-only \
  --batch-size ${BATCH_SIZE} \
  --numa-cores-per-instance ${CORES_PER_INSTANCE} \
  --output-dir ${OUTPUT_DIR} \
  --in-graph ${PRETRAINED_MODEL} \
  ${dataset_arg} \
  $@ \
  -- input_height=224 input_width=224 warmup_steps=500 steps=1000 \
  input_layer='input' output_layer='MobilenetV1/Predictions/Reshape_1'

if [[ $? == 0 ]]; then
  echo "Summary total images/sec:"
  grep 'Average Throughput:' ${OUTPUT_DIR}/mobilenet_v1_int8_inference_bs${BATCH_SIZE}_cores${CORES_PER_INSTANCE}_all_instances.log  | awk -F' ' '{sum+=$3;} END{print sum} '
else
  exit 1
fi
