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
  echo "Please set PRECISION to int8, fp32 or bfloat16."
  exit 1
fi

if [ -z "${PRETRAINED_MODEL}" ]; then
    if [[ $PRECISION == "bfloat16" || $PRECISION == "fp32" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/3dunet_dynamic_ndhwc.pb"
        if [ $PRECISION == "bfloat16" ]; then
            export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_REMOVE="Mul"
        fi
    elif [[ $PRECISION == "int8" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/3dunet_fused_pad_int8.pb"
    else
        echo "The specified precision '${PRECISION}' is unsupported."
        echo "Supported precisions are: int8, fp32 and bfloat16"
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

if [[ $PRECISION == "bfloat16" ]]; then
    BATCH_SIZE="8"
else
    BATCH_SIZE="6"
fi

MODE="inference"
CORES_PER_INSTANCE="socket"
NUM_OF_CORES_PER_SOCKET=$(lscpu | grep "Core(s) per socket" | awk '{split($0,a,":"); print a[2]}')

source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=3d_unet_mlperf \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --numa-cores-per-instance ${CORES_PER_INSTANCE} \
  --data-num-intra-threads ${NUM_OF_CORES_PER_SOCKET} --data-num-inter-threads 1 \
  $@ \
  -- warmup_steps=50 steps=100

if [[ $? == 0 ]]; then
  echo "Throughput samples/sec:"
  cat ${OUTPUT_DIR}/3d_unet_mlperf_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | grep "Throughput:.*samples/sec" | sed -e s"/.*://;s/samples\/sec//"
  exit 0
else
  exit 1
fi
