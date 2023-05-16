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
  echo "Please set PRECISION to int8 or bfloat16."
  exit 1
fi
if [[ $PRECISION != "int8" ]] && [ $PRECISION != "bfloat16" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: int8 and bfloat16"
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

if [ -z "${WARMUP_STEPS}" ]; then
  WARMUP_STEPS="warmup_steps=20"
else
  WARMUP_STEPS="warmup_steps=$WARMUP_STEPS"
fi
echo "WARMUP_STEPS: $WARMUP_STEPS"

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

# Get number of cores per socket line from lscpu
export OMP_NUM_THREADS=4

MODE="inference"

# System envirables  
export TF_ONEDNN_USE_SYSTEM_ALLOCATOR=1

# clean up old log files if found
rm -rf ${OUTPUT_DIR}/distilbert_base_${PRECISION}_bs1_Latency_inference_instance_*


source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_get_socket_cores_lists
_command numactl --localalloc --physcpubind=${cores_per_socket_arr[0]} python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=distilbert_base \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${IN_GRAPH} \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size=1 \
  --num-intra-threads ${cores_per_socket} \
  --num-inter-threads -1 \
  --weight-sharing \
  $@ \
  -- \
  $WARMUP_STEPS >> ${OUTPUT_DIR}/distilbert_base_${PRECISION}_bs1_Latency_inference_instance_0.log 2>&1 & \
numactl --localalloc --physcpubind=${cores_per_socket_arr[1]} python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=distilbert_base \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${IN_GRAPH} \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size=1 \
  --num-intra-threads ${cores_per_socket} \
  --num-inter-threads -1 \
  --weight-sharing \
  $@ \
  -- \
  $WARMUP_STEPS >> ${OUTPUT_DIR}/distilbert_base_${PRECISION}_bs1_Latency_inference_instance_1.log 2>&1 & \
  wait

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/distilbert_base_${PRECISION}_bs1_Latency_inference_instance_*.log | grep "Total aggregated Throughput" | sed -e s"/.*: //"
  echo "Throughput summary:"
  grep 'Total aggregated Throughput' ${OUTPUT_DIR}/distilbert_base_${PRECISION}_bs1_Latency_inference_instance_*.log | awk -F' ' '{sum+=$4;} END{print sum} '
  exit 0
else
  exit 1
fi
