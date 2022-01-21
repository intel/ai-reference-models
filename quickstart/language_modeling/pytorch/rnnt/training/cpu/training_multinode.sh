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

if [ ! -e "${MODEL_DIR}/models/language_modeling/pytorch/rnnt/training/cpu/train.py" ]; then
  echo "Could not find the script of train.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the train.py exist at the: \${MODEL_DIR}/models/language_modeling/pytorch/rnnt/training/cpu/train.py"
  exit 1
fi

if [ ! -d "${DATASET_DIR}/dataset/LibriSpeech" ]; then
  echo "The DATASET_DIR \${DATASET_DIR}/dataset/LibriSpeech does not exist"
  exit 1
fi

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "The OUTPUT_DIR '${OUTPUT_DIR}' does not exist"
  exit 1
fi

if [[ $1 == "avx-fp32" ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""
if [ "$1" == "bf16" ]; then
    ARGS="$ARGS bf16"
    echo "### running bf16 datatype"
elif [[ $1 == "fp32" || $1 == "avx-fp32" ]]; then
    ARGS="$ARGS fp32"
    echo "### running fp32 datatype"
else
    echo "The specified precision '${1}' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, and bf16"
    exit 1
fi

PRECISION=$1

torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
if [[ ! -z "${NUM_STEPS}" ]]; then
    NUM_STEPS=$NUM_STEPS bash ${MODEL_DIR}/models/language_modeling/pytorch/rnnt/training/cpu/train_multinode.sh $ARGS 2>&1 | tee -a ${OUTPUT_DIR}/rnnt-training-distributed-${PRECISION}.log
else
    bash ${MODEL_DIR}/models/language_modeling/pytorch/rnnt/training/cpu/train_multinode.sh $ARGS 2>&1 | tee -a ${OUTPUT_DIR}/rnnt-training-distributed-${PRECISION}.log
fi
