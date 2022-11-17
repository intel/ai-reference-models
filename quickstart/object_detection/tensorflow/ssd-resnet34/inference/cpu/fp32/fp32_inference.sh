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

PRETRAINED_MODEL=${PRETRAINED_MODEL-"$MODEL_DIR/pretrained_model/ssd_resnet34_fp32_bs1_pretrained_model.pb"}

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
    --model-name ssd-resnet34 \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size ${BATCH_SIZE} \
    --socket-id 0 \
    --benchmark-only \
    --in-graph $PRETRAINED_MODEL \
    --model-source-dir $TF_MODELS_DIR \
    --output-dir ${OUTPUT_DIR} \
    $@
