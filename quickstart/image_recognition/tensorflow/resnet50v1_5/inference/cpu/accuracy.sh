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
  echo "Please set PRECISION to fp32 or int8 or bfloat16 or bfloat32."
  exit 1
fi
if [[ $PRECISION != "fp32" ]] && [[ $PRECISION != "int8" ]] && [[ $PRECISION != "bfloat16" ]] && [[ $PRECISION != "bfloat32" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, int8, bfloat16 and bfloat32"
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

if [ -z "${PRETRAINED_MODEL}" ]; then
    if [[ $PRECISION == "int8" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/bias_resnet50.pb"
    elif [[ $PRECISION == "bfloat16" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/bf16_resnet50_v1.pb"
    elif [[ $PRECISION == "fp32" || $PRECISION == "bfloat32" ]]; then
        PRETRAINED_MODEL="${MODEL_DIR}/pretrained_model/resnet50_v1.pb"
    else
        echo "The specified precision '${PRECISION}' is unsupported."
        echo "Supported precisions are: fp32, int8, bfloat16 and bfloat32"
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

# System envirables
export TF_ENABLE_MKL_NATIVE_FORMAT=1
export TF_ONEDNN_ENABLE_FAST_CONV=1

#Set up env variable for bfloat32
if [[ $PRECISION=="bfloat32" ]]; then
  ONEDNN_DEFAULT_FPMATH_MODE=BF16
  PRECISION="fp32"
fi

MODE="inference"

# If batch size env is not mentioned, then the workload will run with the default batch size.
BATCH_SIZE="${BATCH_SIZE:-"100"}"

source "${MODEL_DIR}/quickstart/common/utils.sh"
_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=resnet50v1_5 \
  --precision ${PRECISION} \
  --mode=${MODE} \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --accuracy-only \
  $@ 2>&1 | tee ${OUTPUT_DIR}/resnet50v1_5_${PRECISION}_${MODE}_bs${BATCH_SIZE}_accuracy.log

if [[ $? == 0 ]]; then
  echo "Accuracy summary:"
  cat ${OUTPUT_DIR}/resnet50v1_5_${PRECISION}_${MODE}_bs${BATCH_SIZE}_accuracy.log | grep "Processed 50000 images. (Top1 accuracy, Top5 accuracy)" | sed -e "s/.* = //"
  exit 0
else
  exit 1
fi
