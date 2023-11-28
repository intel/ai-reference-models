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

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32."
  exit 1
fi

if [[ $PRECISION != "fp32" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32."
  exit 1
fi

# Use synthetic data (no --data-location arg) if no DATASET_DIR is set

if [ -z "${USE_REAL_DATA}" ]; then
  echo "Using synthetic data, since the USE_REAL_DATA environment variable is not set."
  USE_REAL_DATA="False"
elif [ "${USE_REAL_DATA}"=="False" ]; then
  USE_REAL_DATA=""
else
  USE_REAL_DATA="True"
fi
dataset_arg="--use_real_data=${USE_REAL_DATA}"

if [[ ! -f "${PRETRAINED_MODEL}" ]]; then
  echo "The file specified by the PRETRAINED_MODEL environment variable (${PRETRAINED_MODEL}) does not exist."
  exit 1
fi

if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="32"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

MODE="inference"

source "${MODEL_DIR}/quickstart/common/utils.sh"

_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=edsr \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  ${dataset_arg} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  $@ \
  warmup_steps=10 steps=50 \
  input_layer='IteratorGetNext' output_layer='NHWC_output'
