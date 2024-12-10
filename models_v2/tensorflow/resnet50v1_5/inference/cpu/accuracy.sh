#!/usr/bin/env bash
#
# Copyright (c) 2021 Intel Corporation
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
  echo "Please set PRECISION to fp32 or int8 or bfloat16 or fp16 or bfloat32."
  exit 1
fi
if [[ $PRECISION != "fp32" ]] && [[ $PRECISION != "int8" ]] && [[ $PRECISION != "bfloat16" ]] && [[ $PRECISION != "fp16" ]] && [[ $PRECISION != "bfloat32" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, int8, bfloat16, fp16 and bfloat32"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${PRETRAINED_MODEL}" ]; then
    if [[ $PRECISION == "int8" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/bias_resnet50.pb"
    elif [[ $PRECISION == "bfloat16" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/bf16_resnet50_v1.pb"
    elif [[ $PRECISION == "fp32" || $PRECISION == "bfloat32" || $PRECISION == "fp16" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/resnet50_v1.pb"
    else
        echo "The specified precision '${PRECISION}' is unsupported."
        echo "Supported precisions are: fp32, int8, bfloat16, fp16 and bfloat32"
        exit 1
    fi
    if [[ ! -f "${PRETRAINED_MODEL}" ]]; then
    echo "The pretrained model could not be found. Please set the PRETRAINED_MODEL env var to point to the frozen graph file."
    exit 1
    fi
elif [[ ! -f "${PRETRAINED_MODEL}" ]]; then
  echo "The file specified by the PRETRAINED_MODEL environment variable (${PRETRAINED_MODEL}) does not exist."
  exit 1
fi

#Set up env variable for bfloat32
if [[ $PRECISION == "bfloat32" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
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
  NUM_INSTANCES=`expr $cores_per_socket / $CORES_PER_NUMA`

  echo "CORES_PER_SOCKET: $CORES_PER_SOCKET"
  echo "SOCKETS: $SOCKETS"
  echo "NUMAS: $NUMAS"
  echo "CORES_PER_INSTANCE: $CORES_PER_INSTANCE"
fi

# If OMP_NUM_THREADS env is not mentioned, then run with the default value
if [ -z "${OMP_NUM_THREADS}" ]; then
  export OMP_NUM_THREADS=${CORES_PER_INSTANCE}
fi

printf '=%.0s' {1..100}
printf "\nSummary of environment variable settings:\n"
# Setting environment variables
if [ -z "${TF_THREAD_PINNING_MODE}" ]; then
  # By default, pinning is none and spinning is enabled
  export TF_THREAD_PINNING_MODE=none,$(($CORES_PER_INSTANCE-1)),400
fi
echo "TF_THREAD_PINNING_MODE=$TF_THREAD_PINNING_MODE"
printf '=%.0s' {1..100}
printf '\n'

# Remove old log file
rm -rf  ${OUTPUT_DIR}/resnet50v1_5_${PRECISION}_${MODE}_bs${BATCH_SIZE}_accuracy.log

source "${MODEL_DIR}/models_v2/common/utils.sh"
_ht_status_spr
_command numactl -N0 -m0 python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=resnet50v1_5 \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --accuracy-only \
  $@ 2>&1 | tee ${OUTPUT_DIR}/resnet50v1_5_${PRECISION}_${MODE}_bs${BATCH_SIZE}_accuracy.log

if [[ $? == 0 ]]; then
  echo "Accuracy summary:"
  cat ${OUTPUT_DIR}/resnet50v1_5_${PRECISION}_${MODE}_bs${BATCH_SIZE}_accuracy.log | grep "Processed 50000 images. (Top1 accuracy, Top5 accuracy)" | sed -e "s/.* = //"
  exit 0
else
  exit 1
fi
