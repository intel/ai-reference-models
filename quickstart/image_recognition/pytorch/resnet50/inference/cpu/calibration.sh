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
if [ ! -e "${MODEL_DIR}/models/image_recognition/pytorch/common/main.py"  ]; then
    echo "Could not find the script of main.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the main.py exist at the: \${MODEL_DIR}/models/image_recognition/pytorch/common/main.py"
    exit 1
fi

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

if [ -z "${CONFIG_FILE}" ]; then
  echo "The required environment variable CONFIG_FILE has not been set"
  echo "Please set CONFIG_FILE to where to store the int8 config json."
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, avx-fp32, int8, avx-int8, or bf16."
  exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

BATCH_SIZE=128

rm -rf ${OUTPUT_DIR}/resnet50_accuracy_log*

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

# download pretrained weight.
python ${MODEL_DIR}/models/image_recognition/pytorch/common/hub_help.py \
    --url https://download.pytorch.org/models/resnet50-0676ba61.pth

ARGS=""
ARGS="$ARGS -e -a resnet50 ${DATASET_DIR}"

echo "running int8 path"
ARGS="$ARGS --int8 --configure-dir ${CONFIG_FILE}"


python -m intel_extension_for_pytorch.cpu.launch \
    --memory-allocator jemalloc \
    ${MODEL_DIR}/models/image_recognition/pytorch/common/main.py \
    $ARGS \
    --ipex \
    --pretrained \
    -j 0 \
    --calibration \
    -b $BATCH_SIZE 2>&1 | tee ${OUTPUT_DIR}/resnet50_accuracy_log_${PRECISION}.log \

wait

accuracy=$(grep 'Accuracy:' ${OUTPUT_DIR}/resnet50_accuracy_log_${PRECISION}.log |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
echo "resnet50;"accuracy";${PRECISION};${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
