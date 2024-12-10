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

echo 'MODEL_DIR='$MODEL_DIR
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'DATASET_DIR='$DATASET_DIR

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
elif [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "bfloat16" ] && [ ${PRECISION} != "bfloat32" ] && [ ${PRECISION} != "fp16" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32 fp16 bfloat32 and bfloat16."
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "DATASET_DIR environment variable is required for running accuracy."
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "DATASET_DIR '${DATASET_DIR}' does not exist."
  exit 1
fi

if [ -z "${DATASET_NAME}" ]; then
  echo "DATASET_NAME environment variable is not set."
  echo "Using default 'squad' dataset."
  DATASET_NAME=squad
fi

source "${MODEL_DIR}/models_v2/common/utils.sh"
_get_numa_cores_lists
echo "Cores per node: ${cores_per_node}"

# If cores per instance env is not mentioned, then the workload will run with the default value.
if [ -z "${CORES_PER_INSTANCE}" ]; then
  CORES_PER_INSTANCE=${cores_per_node}
  echo "Runs an instance per ${CORES_PER_INSTANCE} cores."
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="32"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

if [ -z "${WARMUP_STEPS}" ]; then
  WARMUP_STEPS="10"
fi

if [ -z "${STEPS}" ]; then
  STEPS="30"
fi

if [ -z "${TF_THREAD_PINNING_MODE}" ]; then
  echo "TF_THREAD_PINNING_MODE is not set. Setting it to the following default value:"
  export TF_THREAD_PINNING_MODE=none,$(($CORES_PER_INSTANCE-1)),400
  echo "TF_THREAD_PINNING_MODE: $TF_THREAD_PINNING_MODE"
fi

# set env for Bfloat32
if [[ $PRECISION == "bfloat32" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
  echo "ONEDNN_DEFAULT_FPMATH_MODE: "$ONEDNN_DEFAULT_FPMATH_MODE
fi

_ht_status_spr
_command numactl -N0 -m0 python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=bert_large_hf \
  --dataset-name=${DATASET_NAME} \
  --precision ${PRECISION} \
  --mode=inference \
  --framework tensorflow \
  --output-dir ${OUTPUT_DIR} \
  --data-location=${DATASET_DIR} \
  --batch-size ${BATCH_SIZE} \
  --accuracy-only \
  --verbose 2>&1 | tee ${OUTPUT_DIR}/bert_large_hf_${PRECISION}_inference_accuracy.log

if [[ $? == 0 ]]; then
  echo "Accuracy:"
  cat ${OUTPUT_DIR}/bert_large_hf_${PRECISION}_inference_accuracy.log | grep "f1" | cut -d '-' -f4 | tail -n 1
  cat ${OUTPUT_DIR}/bert_large_hf_${PRECISION}_inference_accuracy.log | grep "exact_match" | cut -d '-' -f4 | tail -n 1
  exit 0
else
  exit 1
fi

