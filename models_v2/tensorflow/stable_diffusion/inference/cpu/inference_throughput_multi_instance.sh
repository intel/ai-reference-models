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

MODELS=${MODELS-$PWD}

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to either fp32, bfloat32, bfloat16, or fp16."
  exit 1
fi
if [ $PRECISION != "fp32" ] && [ $PRECISION != "bfloat32" ] &&
   [ $PRECISION != "bfloat16" ] && [ $PRECISION != "fp16" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions is: fp32, bfloat32, bfloat16, and fp16."
  exit 1
fi

MODE="inference"

# If batch size env is not mentioned, then the workload will run with the default batch size.
BATCH_SIZE="${BATCH_SIZE:-"1"}"

# If number of steps is not mentioned, then the workload will run with the default value.
NUM_STEPS="${NUM_STEPS:-"50"}"

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
if [ -z "${TF_PATTERN_ALLOW_CTRL_DEPENDENCIES}" ]; then
  # By default, setting TF_PATTERN_ALLOW_CTRL_DEPENDENCIES=1 to allow control dependencies to enable more fusions"
  export TF_PATTERN_ALLOW_CTRL_DEPENDENCIES=1
fi
if [ -z "${TF_USE_LEGACY_KERAS}" ]; then
  # By default, setting TF_USE_LEGACY_KERAS=1 to use (legacy) Keras 2
  export TF_USE_LEGACY_KERAS=1
fi
if [ -z "${TF_USE_ADVANCED_CPU_OPS}" ]; then
  # By default, setting TF_USE_ADVANCED_CPU_OPS=1 to enhace the overall performance
  export TF_USE_ADVANCED_CPU_OPS=1
fi
if [ -z "${TF_ONEDNN_ASSUME_FROZEN_WEIGHTS}" ]; then
  # By default, setting TF_ONEDNN_ASSUME_FROZEN_WEIGHTS=1 to perform weight caching as we're using a SavedModel
  export TF_ONEDNN_ASSUME_FROZEN_WEIGHTS=1
fi
if [ -z "${TF_THREAD_PINNING_MODE}" ]; then
  # By default, pinning is none and spinning is enabled
  export TF_THREAD_PINNING_MODE=none,$(($CORES_PER_INSTANCE-1)),400
fi

echo "TF_PATTERN_ALLOW_CTRL_DEPENDENCIES=$TF_PATTERN_ALLOW_CTRL_DEPENDENCIES"
echo "TF_USE_LEGACY_KERAS=$TF_USE_LEGACY_KERAS"
echo "TF_USE_ADVANCED_CPU_OPS=$TF_USE_ADVANCED_CPU_OPS"
echo "TF_ONEDNN_ASSUME_FROZEN_WEIGHTS=$TF_ONEDNN_ASSUME_FROZEN_WEIGHTS"
echo "TF_THREAD_PINNING_MODE=$TF_THREAD_PINNING_MODE"

if [[ $PRECISION == "bfloat16" ]] && [[ "${TF_USE_ADVANCED_CPU_OPS}" == "1" ]]; then
  if [ -z "${TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD}" ]; then
    # Moving Gelu op to INFERLIST as we're using bfloat16 precision
    export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=Gelu
  fi
  echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=$TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD"
fi
if [[ $PRECISION == "fp16" ]]; then
  if [[ -z "${TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD}" ]] && [[ -z "${TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE}" ]]; then
    if [[ "${TF_USE_ADVANCED_CPU_OPS}" == "1" ]]; then
      # Adding Gelu,Mean,Sum,SquaredDifference op to INFERLIST
      export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=Gelu,Mean,Sum,SquaredDifference
    else
      # Adding Mean,Sum,SquaredDifference op to INFERLIST
      export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=Mean,Sum,SquaredDifference
    fi
    export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE=Mean,Sum,SquaredDifference
  fi
  echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=$TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD"
  echo "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE=$TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE"
fi
# Set up env variable for bfloat32
if [[ $PRECISION == "bfloat32" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
  echo "ONEDNN_DEFAULT_FPMATH_MODE=$ONEDNN_DEFAULT_FPMATH_MODE"
fi
printf '=%.0s' {1..100}
printf '\n'

source "${MODELS}/models_v2/common/utils.sh"
_ht_status_spr
_command python ${MODELS}/benchmarks/launch_benchmark.py \
  --model-name=stable_diffusion \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --output-dir ${OUTPUT_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --steps=${NUM_STEPS} \
  --numa-cores-per-instance=${CORES_PER_INSTANCE} \
  $@ \

if [[ $? == 0 ]]; then
  printf "Time taken by different instances:\n"
  cat ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | grep 'Latency:' | sed -e s"/.*: //"
  echo "Latency (min time):"
  cat ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | grep 'Latency:' | sed -e s"/.*: //" | sort -n | head -1
  printf "\nThroughput for different instances:\n"
  cat ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | grep 'Avg Throughput:' | sed -e s"/.*: //"
  echo "Throughput (total):"
  grep 'Avg Throughput' ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | awk -F' ' '{sum+=$3;} END{print sum} '
  exit 0
else
  exit 1
fi

