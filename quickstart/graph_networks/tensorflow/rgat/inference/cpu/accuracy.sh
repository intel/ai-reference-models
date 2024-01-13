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
  echo "Please set PRECISION to either fp32, bfloat16, fp16, or bfloat32."
  exit 1
fi
if [ $PRECISION != "fp32" ] && [ $PRECISION != "bfloat16" ] &&
   [ $PRECISION != "fp16" ] && [ $PRECISION != "bfloat32" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions is: fp32, bfloat16, fp16, bfloat32"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
elif [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${PRETRAINED_MODEL}" ]; then
  echo "Please set the PRETRAINED_MODEL environment variable to point to the directory containing the pretrained model."
  exit 1
elif [[ ! -d "${PRETRAINED_MODEL}" ]]; then
  echo "The directory specified by the PRETRAINED_MODEL environment variable (${PRETRAINED_MODEL}) does not exist."
  exit 1
fi

if [ -z "${GRAPH_SCHEMA_PATH}" ]; then
  echo "Please set the GRAPH_SCHEMA_PATH environment variable to point to the graph schema (*.pbtxt) file."
  exit 1
elif [[ ! -f "${GRAPH_SCHEMA_PATH}" ]]; then
  echo "The file specified by the GRAPH_SCHEMA_PATH environment variable (${GRAPH_SCHEMA_PATH}) does not exist."
  exit 1
fi

MODE="inference"

# If batch size env is not mentioned, then the workload will run with the default batch size.
BATCH_SIZE="${BATCH_SIZE:-"100"}"

# If cores per instance env is not mentioned, then the workload will run with the default value.
if [ -z "${CORES_PER_INSTANCE}" ]; then
  # Get number of cores per instance
  CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
  SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
  NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
  CORES_PER_INSTANCE=`expr $CORES_PER_SOCKET \* $SOCKETS / $NUMAS`
fi

# Setting environment variables
if [ -z "${TF_USE_LEGACY_KERAS}" ]; then
  # By default, setting TF_USE_LEGACY_KERAS=1 to use (legacy) Keras 2
  export TF_USE_LEGACY_KERAS=1
fi
if [ -z "${TF_ONEDNN_ASSUME_FROZEN_WEIGHTS}" ]; then
  # By default, setting TF_ONEDNN_ASSUME_FROZEN_WEIGHTS=1 to perform weight caching as we're using a SavedModel
  export TF_ONEDNN_ASSUME_FROZEN_WEIGHTS=1
fi
if [ -z "${TF_THREAD_PINNING_MODE}" ]; then
  # By default, pinning is none and spinning is enabled
  export TF_THREAD_PINNING_MODE=none,$(($CORES_PER_INSTANCE-1)),400
fi
echo "TF_USE_LEGACY_KERAS=$TF_USE_LEGACY_KERAS"
echo "TF_ONEDNN_ASSUME_FROZEN_WEIGHTS=$TF_ONEDNN_ASSUME_FROZEN_WEIGHTS"
echo "TF_THREAD_PINNING_MODE=$TF_THREAD_PINNING_MODE"

if [[ $PRECISION == "fp16" ]]; then
  export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
  echo "ONEDNN_MAX_CPU_ISA=$ONEDNN_MAX_CPU_ISA"
fi

# Set up env variable for bfloat32
if [[ $PRECISION == "bfloat32" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
  echo "ONEDNN_DEFAULT_FPMATH_MODE=$ONEDNN_DEFAULT_FPMATH_MODE"
fi

# If OMP_NUM_THREADS env is not mentioned, then run with the default value
if [ -z "${OMP_NUM_THREADS}" ]; then
  export OMP_NUM_THREADS=${CORES_PER_INSTANCE}
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=rgat \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --pretrained_model=${PRETRAINED_MODEL} \
  --graph_schema_path=${GRAPH_SCHEMA_PATH} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --accuracy-only \
  $@ 2>&1 | tee ${OUTPUT_DIR}/rgat_${PRECISION}_${MODE}_bs${BATCH_SIZE}_accuracy.log

if [[ $? == 0 ]]; then
  echo "Accuracy summary:"
  cat ${OUTPUT_DIR}/rgat_${PRECISION}_${MODE}_bs${BATCH_SIZE}_accuracy.log | grep "Test accuracy:" | sed -e "s/.* = //"
  exit 0
else
  exit 1
fi
