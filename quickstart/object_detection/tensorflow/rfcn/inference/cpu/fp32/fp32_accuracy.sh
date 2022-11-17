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

if [ -z "${TF_MODELS_DIR}" ]; then
  echo "The required environment variable TF_MODELS_DIR has not been set"
  exit 1
fi

if [ -z "${PRETRAINED_MODEL}" ]; then
  # If the PRETRAINED_MODEL env var is not set, then we are probably running in a workload
  # container or model package, so check for the tar file and extract it, if needed
  pretrained_model_dir="${OUTPUT_DIR}/pretrained_model/rfcn_resnet101_coco_2018_01_28"
  if [ ! -d "${pretrained_model_dir}" ]; then
    if [[ -f "pretrained_model/rfcn_fp32_model.tar.gz" ]]; then
      mkdir -p ${OUTPUT_DIR}/pretrained_model
      tar -C ${OUTPUT_DIR}/pretrained_model/ -xvf pretrained_model/rfcn_fp32_model.tar.gz
    else
      echo "The pretrained model was not found. Please set the PRETRAINED_MODEL var to point to the frozen graph file."
      exit 1
    fi
  fi
  PRETRAINED_MODEL="${pretrained_model_dir}/frozen_inference_graph.pb"
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --model-source-dir ${TF_MODELS_DIR} \
    --data-location ${DATASET_DIR} \
    --in-graph ${PRETRAINED_MODEL} \
    --batch-size ${BATCH_SIZE} \
    --accuracy-only \
    --output-dir ${OUTPUT_DIR} \
    $@ \
    -- split="accuracy_message"

