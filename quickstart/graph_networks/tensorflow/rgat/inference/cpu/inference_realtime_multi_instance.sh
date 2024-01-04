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
  echo "Please set PRECISION to either fp32, bfloat16, or fp16."
  exit 1
fi
if [ $PRECISION != "fp32" ] && [ $PRECISION != "bfloat16" ] &&
   [ $PRECISION != "fp16" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions is: fp32, bfloat16, fp16"
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

if [ -z "${GRAPH_SCHEMA_PATH}" ]; then
  echo "Please set the GRAPH_SCHEMA_PATH environment variable to point to the graph schema (*.pbtxt) file."
  exit 1
elif [[ ! -f "${GRAPH_SCHEMA_PATH}" ]]; then
  echo "The file specified by the GRAPH_SCHEMA_PATH environment variable (${GRAPH_SCHEMA_PATH}) does not exist."
  exit 1
fi

MODE="inference"

# If batch size env is not mentioned, then the workload will run with the default batch size.
BATCH_SIZE="${BATCH_SIZE:-"1"}"

if [ -z "${STEPS}" ]; then
  STEPS="steps=200"
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
  CORES_PER_INSTANCE=4
else
  CORES_PER_INSTANCE=${CORES_PER_INSTANCE}
fi

# Setting environment variables
# use legacy keras 2.x api, keras 3.x not yet supported
export TF_USE_LEGACY_KERAS=1
echo "TF_USE_LEGACY_KERAS=1"
# Assume frozen weight for inference only to enable weight caching with SavedModel to improve perf
export TF_ONEDNN_ASSUME_FROZEN_WEIGHTS=1
echo "TF_ONEDNN_ASSUME_FROZEN_WEIGHTS=1"
# set thread pinning+spinning config. currently pinning is none and spinning is enabled
export TF_THREAD_PINNING_MODE=none,$(($CORES_PER_INSTANCE-1)),400
echo "TF_THREAD_PINNING_MODE: $TF_THREAD_PINNING_MODE"

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=rgat \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --pretrained_model=${PRETRAINED_MODEL} \
  --graph_schema_path=${GRAPH_SCHEMA_PATH} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --num-intra-threads=${CORES_PER_INSTANCE} \
  --num-inter-threads=1 \
  --numa-cores-per-instance=${CORES_PER_INSTANCE} \
  $@ \
  --$STEPS \
  --$WARMUP_STEPS \

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/rgat_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | grep 'Avg Throughput:' | sed -e s"/.*: //"
  echo "Throughput summary:"
  grep 'Avg Throughput' ${OUTPUT_DIR}/rgat_${PRECISION}_${MODE}_bs${BATCH_SIZE}_cores*_all_instances.log | awk -F' ' '{sum+=$3;} END{print sum} '
  exit 0
else
  exit 1
fi
