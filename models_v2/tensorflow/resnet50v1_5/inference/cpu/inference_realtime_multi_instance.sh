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
  echo "Please set PRECISION to fp32, int8, bfloat16, fp16 or bfloat32."
  exit 1
fi
if [[ $PRECISION != "fp32" ]] && [[ $PRECISION != "int8" ]] && [[ $PRECISION != "bfloat16" ]] && [[ $PRECISION != "fp16" ]] && [[ $PRECISION != "bfloat32" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, int8, bfloat16, fp16 and bfloat32"
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

MODE="inference"

# If cores per instance env is not mentioned, then the workload will run with the default value.
if [ -z "${CORES_PER_INSTANCE}" ]; then
  CORES_PER_INSTANCE="4"
  echo "Running with default ${CORES_PER_INSTANCE} cores per instance"
fi

# If OMP_NUM_THREADS env is not mentioned, then run with the default value
if [ -z "${OMP_NUM_THREADS}" ]; then
  export OMP_NUM_THREADS=${CORES_PER_INSTANCE}
fi

#Set up env variable for bfloat32
if [[ $PRECISION == "bfloat32" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
BATCH_SIZE="${BATCH_SIZE:-"1"}"

if [ -z "${STEPS}" ]; then
  STEPS="steps=1500"
else
  STEPS="steps=$STEPS"
fi
echo "STEPS: $STEPS"

if [ -z "${WARMUP_STEPS}" ]; then
  WARMUP_STEPS="warmup_steps=50"
else
  WARMUP_STEPS="warmup_steps=$WARMUP_STEPS"
fi
echo "WARMUP_STEPS: $WARMUP_STEPS"

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
rm -rf  ${OUTPUT_DIR}/resnet50v1_5_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log

source "${MODEL_DIR}/models_v2/common/utils.sh"
_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=resnet50v1_5 \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  ${dataset_arg} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --numa-cores-per-instance ${CORES_PER_INSTANCE} \
  --num-cores=${CORES_PER_INSTANCE} \
  --num-intra-threads ${CORES_PER_INSTANCE} --num-inter-threads 1 \
  --data-num-intra-threads ${CORES_PER_INSTANCE} --data-num-inter-threads 1 \
  $@ \
  -- \
  $WARMUP_STEPS \
  $STEPS

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/resnet50v1_5_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | grep Throughput: | sed -e s"/.*: //"
  echo "Throughput summary:"
  grep 'Throughput' ${OUTPUT_DIR}/resnet50v1_5_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | awk -F' ' '{sum+=$2;} END{print sum} '
  exit 0
else
  exit 1
fi
