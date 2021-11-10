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
TRAINING_EPOCHS=${TRAINING_EPOCHS:-1}

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
  echo "Please set PRECISION to fp32, avx-fp32, or bf16."
  exit 1
fi

cd ${MODEL_DIR}/models/resnet50/examples/imagenet

# download pretrained weight.
python hub_help.py --url https://download.pytorch.org/models/resnet50-0676ba61.pth

export work_space=${OUTPUT_DIR}
export TRAINING_EPOCHS=${TRAINING_EPOCHS}

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ $PRECISION == "bf16" ]]; then
    bash run_training_ipex_spr.sh resnet50 $DATASET_DIR bf16 2>&1 | tee -a ${OUTPUT_DIR}/resnet50-training-bf16.log
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    bash run_training_ipex_spr.sh resnet50 $DATASET_DIR fp32 2>&1 | tee -a ${OUTPUT_DIR}/resnet50-training-fp32.log
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, and bf16"
    exit 1
fi
