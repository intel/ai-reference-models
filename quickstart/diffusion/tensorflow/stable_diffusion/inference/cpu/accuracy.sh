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

# Setting environment variables
if [ -z "${TF_PATTERN_ALLOW_CTRL_DEPENDENCIES}" ]; then
  echo "TF_PATTERN_ALLOW_CTRL_DEPENDENCIES is not set. By default, setting TF_PATTERN_ALLOW_CTRL_DEPENDENCIES=1 to allow control dependencies for fusions"
  export TF_PATTERN_ALLOW_CTRL_DEPENDENCIES=1
fi
if [ -z "${TF_USE_LEGACY_KERAS}" ]; then
  echo "TF_USE_LEGACY_KERAS is not set. By default, setting TF_USE_LEGACY_KERAS=1 to use Keras 2"
  export TF_USE_LEGACY_KERAS=1
fi
if [ -z "${TF_USE_ADVANCED_CPU_OPS}" ]; then
  echo "TF_USE_ADVANCED_CPU_OPS is not set. By default, setting TF_USE_ADVANCED_CPU_OPS=1 to enhace the overall performance"
  export TF_USE_ADVANCED_CPU_OPS=1
fi
if [ -z "${TF_ONEDNN_ASSUME_FROZEN_WEIGHTS}" ]; then
  echo "TF_ONEDNN_ASSUME_FROZEN_WEIGHTS is not set. By default, setting TF_ONEDNN_ASSUME_FROZEN_WEIGHTS=1 to perform weight caching as we're using a SavedModel"
  export TF_ONEDNN_ASSUME_FROZEN_WEIGHTS=1
fi

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
if [ $PRECISION == "bfloat16" ]; then
  echo "Moving Gelu op to INFERLIST as we're using bfloat16 precision"
  export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=Gelu
fi
if [ $PRECISION == "fp16" ]; then
  echo "Adding Gelu,Mean,Sum,SquaredDifference op to INFERLIST as we're using fp16 precision"
  export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_ADD=Gelu,Mean,Sum,SquaredDifference
  export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE=Mean,Sum,SquaredDifference
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
elif [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

MODE="inference"

# If batch size env is not mentioned, then the workload will run with the default batch size.
BATCH_SIZE="${BATCH_SIZE:-"1"}"

# If number of steps is not mentioned, then the workload will run with the default value.
NUM_STEPS="${NUM_STEPS:-"50"}"

source "${MODELS}/quickstart/common/utils.sh"
_command python ${MODELS}/benchmarks/launch_benchmark.py \
  --model-name=stable_diffusion \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --steps=${NUM_STEPS} \
  --data-location ${DATASET_DIR} \
  --accuracy-only \
  $@ 2>&1 | tee ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_${MODE}_bs${BATCH_SIZE}_accuracy.log

if [[ $? == 0 ]]; then
  echo "FID summary:"
  cat ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_${MODE}_bs${BATCH_SIZE}_accuracy.log | grep "FID score between original and generated images:" | sed -e "s/.* = //"
  exit 0
else
  exit 1
fi
