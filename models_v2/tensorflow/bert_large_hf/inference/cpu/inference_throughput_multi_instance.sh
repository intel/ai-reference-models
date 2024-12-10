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
elif [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "bfloat16" ] && [ ${PRECISION} != "bfloat32" ] && [ ${PRECISION} != "fp16" ]; then
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

source "${MODEL_DIR}/models_v2/common/utils.sh"
_get_numa_cores_lists
echo "Cores per node: ${cores_per_node}"

# If cores per instance env is not mentioned, run with default value.
if [ -z "${CORES_PER_INSTANCE}" ]; then
  CORES_PER_INSTANCE=${cores_per_node}
  echo "Runs an instance per ${CORES_PER_INSTANCE} cores."
fi

# If OMP_NUM_THREADS env is not mentioned, then run with the default value.
if [ -z "${OMP_NUM_THREADS}" ]; then
  export OMP_NUM_THREADS=${CORES_PER_INSTANCE}
else
  export OMP_NUM_THREADS=${OMP_NUM_THREADS}
fi

# If batch size env is not mentioned, run with default batch size.
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="128"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

if [ -z "${WARMUP_STEPS}" ]; then
  WARMUP_STEPS="10"
fi

if [ -z "${STEPS}" ]; then
  STEPS="50"
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
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=bert_large_hf \
  --dataset-name=${DATASET_NAME} \
  --precision ${PRECISION} \
  --mode=inference \
  --framework tensorflow \
  --output-dir ${OUTPUT_DIR} \
  ${dataset_dir} \
  --batch-size ${BATCH_SIZE} \
  --numa-cores-per-instance ${CORES_PER_INSTANCE} \
  --num-cores=${CORES_PER_INSTANCE} \
  --num-intra-threads ${CORES_PER_INSTANCE} \
  --num-inter-threads 1 \
  --warmup-steps=${WARMUP_STEPS} \
  --steps=${STEPS} \
  --benchmark-only \
  --verbose

if [[ $? == 0 ]]; then
  echo "Throughput summary:"
  grep "Throughput" ${OUTPUT_DIR}/bert_large_hf_${PRECISION}_inference_bs${BATCH_SIZE}_cores*_all_instances.log | awk ' {sum+=$(NF);} END{print sum} '
  exit 0
else
  exit 1
fi
