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

if [ -z "${CHECKPOINT_DIR}" ]; then
  echo "The required Checkpoint directory has not been set."
  echo "Please set a directory where the model and data will be downloaded."
  exit 1
fi

if [ ! -d "${CHECKPOINT_DIR}" ]; then
  echo "The CHECKPOINT_DIR '${CHECKPOINT_DIR}' does not exist. The script will download the saved model."
  mkdir -p ${CHECKPOINT_DIR}
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, bfloat16 or fp16."
  exit 1
elif [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "bfloat16" ] && [ ${PRECISION} != "fp16" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, bfloat16 and fp16"
  exit 1
fi

cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
export OMP_NUM_THREADS=${cores_per_socket}

if [ -z "${TF_USE_LEGACY_KERAS}" ]; then
  source "${MODEL_DIR}/quickstart/language_modeling/tensorflow/gpt_j/inference/cpu/apply.sh"
else
  echo "Legacy keras is used"
  source "${MODEL_DIR}/quickstart/language_modeling/tensorflow/gpt_j/inference/cpu/apply_legacy_keras.sh"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"

_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=gpt_j \
  --precision=${PRECISION} \
  --mode=inference \
  --accuracy-only \
  --framework tensorflow \
  --output-dir ${OUTPUT_DIR} \
  --checkpoint=${CHECKPOINT_DIR} \
  --num-intra-threads ${cores_per_socket} \
  --num-inter-threads 1 \
  $@
