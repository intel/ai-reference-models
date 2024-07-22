#!/usr/bin/env bash
#
# Copyright (c) 2024 Intel Corporation
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
  echo "The required environment variable OUTPUT_DIR has not been set."
  exit 1
fi

# Delete existing output directory and create a new one
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set."
  echo "Please set PRECISION to fp32/fp16/bf16/bf32."
  exit 1
elif [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "bfloat16" ] && [ ${PRECISION} != "fp16" ] && [ ${PRECISION} != "bfloat32" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32 fp16 bfloat32 and bfloat16."
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "DATASET_DIR environment variable is not set."
  echo "Model script will download 'bert-large-uncased-whole-word-masking' model from huggingface.co/models."
  dataset_dir=""
else
  dataset_dir=" --data-location=${DATASET_DIR}"
fi

if [ -z "${DATASET_NAME}" ]; then
  echo "DATASET_NAME environment variable is not set."
  echo "Using default 'squad' dataset."
  DATASET_NAME=squad
fi

# set env for Bfloat32
if [[ $PRECISION == "bfloat32" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
  echo "ONEDNN_DEFAULT_FPMATH_MODE: "$ONEDNN_DEFAULT_FPMATH_MODE
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

if [ -z "${WARMUP_STEPS}" ]; then
  WARMUP_STEPS="10"
fi

if [ -z "${STEPS}" ]; then
  STEPS="30"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_get_numa_cores_lists
echo "Cores per node: ${cores_per_node}"

_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=bert_large_hf \
  --dataset-name=${DATASET_NAME} \
  --precision ${PRECISION} \
  --socket-id 0 \
  --mode=inference \
  --framework tensorflow \
  --output-dir ${OUTPUT_DIR} \
  ${dataset_dir} \
  --batch-size ${BATCH_SIZE} \
  --warmup-steps=${WARMUP_STEPS} \
  --num-inter-threads=${cores_per_node} \
  --num-intra-threads=${cores_per_node} \
  --steps=${STEPS} \
  --benchmark-only \
  --verbose
