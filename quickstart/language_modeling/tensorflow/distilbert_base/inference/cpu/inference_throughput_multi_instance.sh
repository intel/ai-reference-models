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
  echo "Please set PRECISION to fp32, fp16, bfloat32, bfloat16 or int8"
  exit 1
fi

if [ -z "${WARMUP_STEPS}" ]; then
  echo "Setting WARMUP_STEPS to 20"
  WARMUP_STEPS="20"
fi

if [ -z "${STEPS}" ]; then
  echo "Setting STEPS to 100"
  STEPS=100
fi

if [ $PRECISION != "fp32" ] && [ $PRECISION != "int8" ] &&
   [ $PRECISION != "bfloat16" ] && [ $PRECISION != "fp16" ] &&
   [ $PRECISION != "bfloat32" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, fp16, bfloat32, bfloat16 and int8"
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
if [[ $PRECISION == "bfloat16" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-"198"}"
elif [[ $PRECISION == "int8" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-"110"}"
else 
  BATCH_SIZE="${BATCH_SIZE:-"56"}"
fi

# If cores per instance env is not mentioned, then the workload will run with the default value.
if [ -z "${CORES_PER_INSTANCE}" ]; then
  # Get number of cores per instance
  CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
  SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
  NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
  CORES_PER_INSTANCE=`expr $CORES_PER_SOCKET \* $SOCKETS / $NUMAS`

  echo "CORES_PER_SOCKET: $CORES_PER_SOCKET"
  echo "SOCKETS: $SOCKETS"
  echo "NUMAS: $NUMAS"
  echo "CORES_PER_INSTANCE: $CORES_PER_INSTANCE"
fi

# If OMP_NUM_THREADS env is not mentioned, then run with the default value
if [ -z "${OMP_NUM_THREADS}" ]; then 
  export OMP_NUM_THREADS=${CORES_PER_INSTANCE}
fi
# Setting environment variables
if [ -z "${TF_USE_ADVANCED_CPU_OPS}" ]; then
  # By default, setting TF_USE_ADVANCED_CPU_OPS=1 to enhace the overall performance
  export TF_USE_ADVANCED_CPU_OPS=1
fi

if [ -z "${TF_THREAD_PINNING_MODE}" ]; then
  # By default, pinning is none and spinning is enabled
  export TF_THREAD_PINNING_MODE=none,$(($CORES_PER_INSTANCE-1)),400
fi

printf '=%.0s' {1..100}
printf "\nSummary of environment variable settings:\n"
echo "TF_USE_ADVANCED_CPU_OPS=$TF_USE_ADVANCED_CPU_OPS"
echo "TF_THREAD_PINNING_MODE=$TF_THREAD_PINNING_MODE"

if [[ $PRECISION == "bfloat16" || $PRECISION == "fp16" ]]; then
  if [[ -z "${TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD}" ]] && [[ -z "${TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE}" ]]; then
      # Adding Mean op to INFERLIST
    export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=Mean
    export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE=Mean
  fi
  echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=$TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD"
  echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE=$TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE"
  if [[ $PRECISION == "fp16" ]]; then
    export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
    echo "ONEDNN_MAX_CPU_ISA=$ONEDNN_MAX_CPU_ISA"
  fi
fi
# Set up env variable for bfloat32
if [[ $PRECISION == "bfloat32" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
  echo "ONEDNN_DEFAULT_FPMATH_MODE=$ONEDNN_DEFAULT_FPMATH_MODE"
fi
printf '=%.0s' {1..100}
printf '\n'

source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_get_numa_cores_lists
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
         --num-intra-threads=${CORES_PER_INSTANCE} \
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

