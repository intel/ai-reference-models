#!/bin/bash

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
CORES_PER_INSTANCE="socket"
cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, fp16, bfloat16 or int8"
  exit 1
fi

if [ -z "${WARMUP_STEPS}" ]; then
  echo "Setting WARMUP_STEPS to 10"
  WARMUP_STEPS="10"
fi

if [ -z "${STEPS}" ]; then
  echo "Setting STEPS to 50"
  STEPS=50
fi

if [ $PRECISION != "fp32" ] && [ $PRECISION != "int8" ] &&
   [ $PRECISION != "bfloat16" ] && [ $PRECISION != "fp16"]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, fp16, bfloat16 and int8"
  exit 1
fi

if [ ! -z "${IN_GRAPH}" ]; then
    if [ ! -f ${IN_GRAPH} ]; then
        echo "The frozen graph could not be found"
        exit 1
    fi
else
    echo "The required environment variable IN_GRAPH has not been set"
    echo "Set it to the path for the frozen graph"
    exit 1
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
BATCH_SIZE="${BATCH_SIZE:-"56"}"

source "${MODEL_DIR}/quickstart/common/utils.sh"
_get_numa_cores_lists
echo "Cores per node: ${cores_per_node}"

# Setting environment variables
echo "Advanced settings for improved performance: "
echo "Setting TF_USE_ADVANCED_CPU_OPS to 1, to enhace the overall performance"
export TF_USE_ADVANCED_CPU_OPS=1
echo "TF_USE_ADVANCED_CPU_OPS = ${TF_USE_ADVANCED_CPU_OPS}"

if [[ ${TF_USE_ADVANCED_CPU_OPS} == "1" ]]; then
	if [[ $PRECISION == "bfloat16" ]]; then
		echo "TF_USE_ADVANCED_CPU_OPS is on for bfloat16 precision"
		export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=Mean
    export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE=Mean
		echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD = ${TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD}"
	elif [[ $PRECISION == "fp16" ]]; then
		echo "TF_USE_ADVANCED_CPU_OPS is on for fp16 precision"
		export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=Mean
		export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE=Mean
    export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
		echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD = ${TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD}"
		echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE = ${TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE}"
    echo "ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX_FP16 = ${ONEDNN_MAX_CPU_ISA}"
  fi
fi

echo "Configuring thread pinning and spinning settings"
export TF_THREAD_PINNING_MODE=none,$((${cores_per_node} - 1)),400
echo "TF_THREAD_PINNING_MODE: $TF_THREAD_PINNING_MODE"

_ht_status_spr
_command python benchmarks/launch_benchmark.py \
         --model-name=distilbert_base \
         --precision=${PRECISION} \
         --mode=inference \
         --framework=tensorflow \
         --in-graph=${IN_GRAPH} \
         --data-location=${DATASET_DIR} \
         --benchmark-only \
         --batch-size=${BATCH_SIZE} \
         --output-dir=${OUTPUT_DIR} \
         --num-intra-threads=${cores_per_socket} \
         --num-inter-threads=1 \
         --numa-cores-per-instance=${CORES_PER_INSTANCE} \
         --warmup-steps=${WARMUP_STEPS} \
         --steps=${STEPS} \
         $@

if [[ $? == 0 ]]; then
  grep "Throughput: " ${OUTPUT_DIR}/distilbert_base_${PRECISION}_inference_bs${BATCH_SIZE}_cores${CORES_PER_INSTANCE}_all_instances.log | sed -e "s/.*://;s/ms//" | awk ' {sum+=$(1);} END{print sum} '
  exit 0
else
  exit 1
fi

