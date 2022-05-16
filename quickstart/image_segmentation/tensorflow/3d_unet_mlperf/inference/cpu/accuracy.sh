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
  echo "Please set PRECISION to int8, fp32 or bfloat16."
  exit 1
fi

if [ -z "${PRETRAINED_MODEL}" ]; then
    if [[ $PRECISION == "bfloat16" || $PRECISION == "fp32" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/3dunet_dynamic_ndhwc.pb"
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

MODE="inference"
BATCH_SIZE="1"
source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=3d_unet_mlperf \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --accuracy-only \
  $@ \
  -- DEBIAN_FRONTEND=noninteractive 2>&1 | tee ${OUTPUT_DIR}/3d_unet_mlperf_${PRECISION}_${MODE}_bs${BATCH_SIZE}_accuracy.log

if [[ $? == 0 ]]; then
  echo "Accuracy:"
  cat ${OUTPUT_DIR}/3d_unet_mlperf_${PRECISION}_${MODE}_bs${BATCH_SIZE}_accuracy.log | grep "Accuracy:" | head -n 1 | sed -e "s/.*Accuracy: mean = //"
  exit 0
else
  exit 1
fi
