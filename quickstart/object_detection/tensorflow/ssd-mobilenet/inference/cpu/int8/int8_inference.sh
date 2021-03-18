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

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -f "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

# Untar pretrained model files
pretrained_model="ssdmobilenet_int8_pretrained_model_combinedNMS_s8.pb"
if [ ! -f "${pretrained_model}" ]; then
    echo "Following ${pretrained_model} frozen graph file does not exists"
    exit 1
fi
FROZEN_GRAPH="$(pwd)/${pretrained_model}"

source "$(dirname $0)/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
    --data-location ${DATASET_DIR} \
    --in-graph ${FROZEN_GRAPH} \
    --output-dir ${OUTPUT_DIR} \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --socket-id 0 \
    --benchmark-only \
    --batch-size 1 \
    $@

