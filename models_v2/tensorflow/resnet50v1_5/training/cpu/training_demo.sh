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
  echo "Please set PRECISION to fp32, bfloat16 or fp16"
  exit 1
fi

if [[ $PRECISION != "fp32" ]] && [[ $PRECISION != "bfloat16" ]] && [[ $PRECISION != "fp16" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, bfloat16 and fp16"
  exit 1
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="16"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

source "${MODEL_DIR}/models_v2/common/utils.sh"
_command python benchmarks/launch_benchmark.py \
         --model-name=resnet50v1_5 \
         --precision ${PRECISION} \
         --mode=training \
         --framework tensorflow \
         --batch-size ${BATCH_SIZE} \
         --checkpoint ${OUTPUT_DIR} \
         --data-location=${DATASET_DIR} \
         --output-dir ${OUTPUT_DIR} \
         --noinstall \
         $@ \
         -- steps=50 train_epochs=1 epochs_between_evals=1

