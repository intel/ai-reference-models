#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
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

if [ -z "${PRETRAINED_MODEL}" ]; then
  # If no PRETRAINED_MODEL var is set, we are probably running in a workload container
  # or model package, so check for the frozen graph file
  pretrained_model="pretrained_model/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb"
  if [ ! -f "${pretrained_model}" ]; then
    echo "The pretrained model could not be found at: ${pretrained_model}."
    echo "Set the PRETRAINED_MODEL environment variable to the path to the frozen graph file."
    exit 1
  fi
  PRETRAINED_MODEL="${MODEL_DIR}/${pretrained_model}"
fi

CORES_PER_INSTANCE="4"

# If batch size env is not mentioned, then the workload will run with the default batch size which gives the best performance
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="1"
  echo "Running with default/best batch size of ${BATCH_SIZE}"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --in-graph ${PRETRAINED_MODEL} \
  --output-dir ${OUTPUT_DIR} \
  --model-name ssd-mobilenet \
  --framework tensorflow \
  --precision fp32 \
  --mode inference \
  --batch-size=${BATCH_SIZE} \
  --numa-cores-per-instance ${CORES_PER_INSTANCE} \
  --benchmark-only \
  $@

if [[ $? == 0 ]]; then
  echo "Summary total samples/sec:"
  grep 'Total samples/sec' ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs${BATCH_SIZE}_cores${CORES_PER_INSTANCE}_all_instances.log  | awk -F' ' '{sum+=$3;} END{print sum} '
else
  exit 1
fi
