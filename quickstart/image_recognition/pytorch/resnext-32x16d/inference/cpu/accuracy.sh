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
    echo "From which the inference.py exist at the: \${MODEL_DIR}/models/image_recognition/pytorch/common/hub_help.py"
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

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, avx-fp32, int8, avx-int8, or bf16."
  exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

BATCH_SIZE=128

rm -rf ${OUTPUT_DIR}/resnext101_accuracy_log*

# download pretrained weight.
python ${MODEL_DIR}/models/image_recognition/pytorch/common/hub_help.py \
    --url https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth

ARGS=""
ARGS="$ARGS -e -a resnext101_32x16d_wsl --hub ${DATASET_DIR}"

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ $PRECISION == "int8" || $PRECISION == "avx-int8" ]]; then
    echo "running int8 path"
    ARGS="$ARGS --int8 --configure-dir ${MODEL_DIR}/models/image_recognition/pytorch/common/resnext101_configure_sym.json"
elif [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16 --jit"
    echo "running bf16 path"
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    ARGS="$ARGS --jit"
    echo "running fp32 path"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, int8, and avx-int8"
    exit 1
fi

python -m intel_extension_for_pytorch.cpu.launch \
    --use_default_allocator \
    ${MODEL_DIR}/models/image_recognition/pytorch/common/main.py \
    $ARGS \
    --ipex \
    --pretrained \
    -j 0 \
    -b $BATCH_SIZE 2>&1 | tee ${OUTPUT_DIR}/resnext101_accuracy_log_${PRECISION}.log

wait

accuracy=$(grep 'Accuracy:' ${OUTPUT_DIR}/resnext101_accuracy_log_${PRECISION}.log |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
echo "resnext101;"accuracy";${PRECISION};${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
