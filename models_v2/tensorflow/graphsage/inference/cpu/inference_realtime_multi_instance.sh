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
CORES_PER_INSTANCE=4

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to either fp32, bfloat16, fp16, int8, or bfloat32"
  exit 1
fi
if [ $PRECISION != "fp32" ] && [ $PRECISION != "bfloat16" ] &&
   [ $PRECISION != "fp16" ] && [ $PRECISION != "int8" ] && [ $PRECISION != "bfloat32" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions is: fp32, bfloat16, fp16, int8, bfloat32"
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


# If batch size env is not mentioned, then the workload will run with the default batch size.
BATCH_SIZE="${BATCH_SIZE:-"1"}"

MODE="inference"

if [ -z "${STEPS}" ]; then
  STEPS="steps=20"
else
  STEPS="steps=$STEPS"
fi
echo "STEPS: $STEPS"

if [ -z "${WARMUP_STEPS}" ]; then
  WARMUP_STEPS="warmup-steps=10"
else
  WARMUP_STEPS="warmup-steps=${WARMUP_STEPS}"
fi
echo "WARMUP_STEPS: ${WARMUP_STEPS}"

# If cores per instance env is not mentioned, then the workload will run with the default value.
if [ -z "${CORES_PER_INSTANCE}" ]; then
  CORES_PER_INSTANCE="4"
  echo "Running with default ${CORES_PER_INSTANCE} cores per instance"
fi

# If OMP_NUM_THREADS env is not mentioned, Get number of cores per instance
if [ -z "${OMP_NUM_THREADS}" ]; then
  export OMP_NUM_THREADS=4
else
  export OMP_NUM_THREADS=${OMP_NUM_THREADS}
fi

if [ -z "${TF_THREAD_PINNING_MODE}" ]; then
  echo "Configuring thread pinning and spinning settings"
  export TF_THREAD_PINNING_MODE=none,$(($CORES_PER_INSTANCE-1)),400
  echo "TF_THREAD_PINNING_MODE: $TF_THREAD_PINNING_MODE"
fi

if [[ $PRECISION == "bfloat32" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
  echo "ONEDNN_DEFAULT_FPMATH_MODE: "$ONEDNN_DEFAULT_FPMATH_MODE
fi

# Get number of cores per socket line from lscpu
cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
cores_per_socket="${cores_per_socket//[[:blank:]]/}"

echo "CORES_PER_INSTANCE: $CORES_PER_INSTANCE"
source "${MODEL_DIR}/models_v2/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=graphsage \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --pretrained-model=${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --num-intra-threads=${CORES_PER_INSTANCE} \
  --num-inter-threads=1 \
  --numa-cores-per-instance=${CORES_PER_INSTANCE} \
  $@ \
  -- \
  $WARMUP_STEPS \
  $STEPS \

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/graphsage_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | grep 'Throughput:' | sed -e s"/.*: //"
  echo "Throughput summary:"
  grep 'Throughput' ${OUTPUT_DIR}/graphsage_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | awk -F' ' '{sum+=$2;} END{print sum} '
  exit 0
else
  exit 1
fi
