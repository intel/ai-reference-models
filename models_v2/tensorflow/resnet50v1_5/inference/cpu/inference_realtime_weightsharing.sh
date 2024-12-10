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
  echo "Please set PRECISION to int8 or bfloat16."
  exit 1
fi
if [[ $PRECISION != "int8" ]] && [[ $PRECISION != "bfloat16" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: bfloat16, and int8"
  exit 1
fi

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
    if [[ $PRECISION == "int8" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/bias_resnet50.pb"
    elif [[ $PRECISION == "bfloat16" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/bf16_resnet50_v1.pb"
    elif [[ $PRECISION == "fp32" || $PRECISION == "bfloat32" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/resnet50_v1.pb"
    else
        echo "The specified precision '${PRECISION}' is unsupported."
        echo "Supported precisions are: fp32, bfloat16, bfloat32 and int8"
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

cores_per_socket="4"

# If OMP_NUM_THREADS env is not mentioned, then run with the default value
if [ -z "${OMP_NUM_THREADS}" ]; then
  export OMP_NUM_THREADS=4
else
  export OMP_NUM_THREADS=${OMP_NUM_THREADS}
fi

MODE="inference"

#Set up env variable for bfloat32
if [[ $PRECISION == "bfloat32" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
fi

# If batch size env is not set, then the workload will run with the default batch size.
BATCH_SIZE="${BATCH_SIZE:-"1"}"

if [ -z "${STEPS}" ]; then
  if [[ $PRECISION == "int8" || $PRECISION == "bfloat16" ]]; then
    STEPS="steps=1500"
  fi
else
  STEPS="steps=$STEPS"
fi
echo "STEPS: $STEPS"

if [ -z "${WARMUP_STEPS}" ]; then
  if [[ $PRECISION == "int8" || $PRECISION == "bfloat16" ]]; then
    WARMUP_STEPS="warmup_steps=100"
  fi
else
  WARMUP_STEPS="warmup_steps=$WARMUP_STEPS"
fi
echo "WARMUP_STEPS: $WARMUP_STEPS"

# System envirables
export TF_ENABLE_MKL_NATIVE_FORMAT=1
export TF_ONEDNN_ENABLE_FAST_CONV=1
export TF_ONEDNN_USE_SYSTEM_ALLOCATOR=1

# clean up old log files if found
rm -rf ${OUTPUT_DIR}/ResNet_50_v1_5_${PRECISION}_bs${BATCH_SIZE}_Latency_inference_instance_*

source "${MODEL_DIR}/models_v2/common/utils.sh"
_ht_status_spr
_get_socket_cores_lists
_command numactl --localalloc --physcpubind=${cores_per_socket_arr[0]} python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=resnet50v1_5 \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  ${dataset_arg} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --num-intra-threads ${cores_per_socket} --num-inter-threads -1 \
  --data-num-intra-threads ${cores_per_socket} --data-num-inter-threads -1 \
  --weight-sharing \
  $@ \
  -- \
  $WARMUP_STEPS \
  $STEPS >> ${OUTPUT_DIR}/ResNet_50_v1_5_${PRECISION}_bs${BATCH_SIZE}_Latency_inference_instance_0.log 2>&1 & \
numactl --localalloc --physcpubind=${cores_per_socket_arr[1]} python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=resnet50v1_5 \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  ${dataset_arg} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --num-intra-threads ${cores_per_socket} --num-inter-threads -1 \
  --data-num-intra-threads ${cores_per_socket} --data-num-inter-threads -1 \
  --weight-sharing \
  $@ \
  -- \
  $WARMUP_STEPS \
  $STEPS >> ${OUTPUT_DIR}/ResNet_50_v1_5_${PRECISION}_bs${BATCH_SIZE}_Latency_inference_instance_1.log 2>&1 & \
  wait

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/ResNet_50_v1_5_${PRECISION}_bs${BATCH_SIZE}_Latency_inference_instance_*.log | grep "Total aggregated Throughput" | sed -e s"/.*: //"
  echo "Throughput summary:"
  grep 'Total aggregated Throughput' ${OUTPUT_DIR}/ResNet_50_v1_5_${PRECISION}_bs${BATCH_SIZE}_Latency_inference_instance_*.log | awk -F' ' '{sum+=$4;} END{print sum} '
  exit 0
else
  exit 1
fi
