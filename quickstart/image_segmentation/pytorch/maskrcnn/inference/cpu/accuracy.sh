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

echo "PRECISION: ${PRECISION}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, avx-fp32, or bf16."
  exit 1
fi

cd ${MODEL_DIR}/models/maskrcnn/maskrcnn-benchmark

export work_space=${OUTPUT_DIR}

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ $PRECISION == "bf16" ]]; then
    bash run_inference_cpu_accuracy.sh bf16 jit 2>&1 | tee -a ${OUTPUT_DIR}/maskrcnn-inference-accuracy-bf16.log
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    bash run_inference_cpu_accuracy.sh fp32 jit 2>&1 | tee -a ${OUTPUT_DIR}/maskrcnn-inference-accuracy-fp32.log
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, and bf16"
    exit 1
fi
