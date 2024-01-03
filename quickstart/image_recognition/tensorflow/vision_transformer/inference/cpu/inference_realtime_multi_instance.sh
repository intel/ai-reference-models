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
  echo "Please set PRECISION to fp32, bfloat16, fp16 or int8."
  exit 1
fi
if [ $PRECISION != "fp32" ] && [ $PRECISION != "bfloat16" ] &&
   [ $PRECISION != "fp16" ] && [ $PRECISION != "int8" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions is: fp32, bfloat16, fp16, int8"
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
  echo "The pretrained model could not be found. Please set the PRETRAINED_MODEL env var to point to the frozen graph file."
  exit 1
elif [[ ! -f "${PRETRAINED_MODEL}" ]]; then
  echo "The file specified by the PRETRAINED_MODEL environment variable (${PRETRAINED_MODEL}) does not exist."
  exit 1
fi

MODE="inference"

# If batch size env is not mentioned, then the workload will run with the default batch size.
BATCH_SIZE="${BATCH_SIZE:-"1"}"

if [ -z "${STEPS}" ]; then
  STEPS="steps=100"
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
echo "CORES_PER_INSTANCE: $CORES_PER_INSTANCE"

# Setting environment variables
echo "Advanced settings for improved performance : "
echo "Setting TF_USE_ADVANCED_CPU_OPS to 1, to enhace the overall performance"
export TF_USE_ADVANCED_CPU_OPS=1
echo "TF_USE_ADVANCED_CPU_OPS = ${TF_USE_ADVANCED_CPU_OPS}"

if [[ ${TF_USE_ADVANCED_CPU_OPS} == "1" ]]; then
	if [[ $PRECISION == "bfloat16" ]]; then
		echo "TF_USE_ADVANCED_CPU_OPS is on for bfloat16 precision"
		export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=Gelu
		echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD = ${TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD}"
	elif [[ $PRECISION == "fp16" ]]; then
		echo "TF_USE_ADVANCED_CPU_OPS is on for fp16 precision"
		export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=Mean,Gelu
		export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE=Mean
		echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD = ${TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD}"
		echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE = ${TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE}"
	fi
fi

echo "Configuring thread pinning and spinning settings"
export TF_THREAD_PINNING_MODE=none,$(($CORES_PER_INSTANCE-1)),400
echo "TF_THREAD_PINNING_MODE: $TF_THREAD_PINNING_MODE"

# Get number of cores per socket line from lscpu
cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
cores_per_socket="${cores_per_socket//[[:blank:]]/}"

source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=vision_transformer \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  ${dataset_arg} \
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
  grep "Throughput: " ${OUTPUT_DIR}/vision_transformer_${PRECISION}_inference_bs1_cores${CORES_PER_INSTANCE}_all_instances.log | sed -e "s/.*://;s/ms//" | awk ' {sum+=$1;} END{print sum} '
  exit 0
else
  exit 1
fi
