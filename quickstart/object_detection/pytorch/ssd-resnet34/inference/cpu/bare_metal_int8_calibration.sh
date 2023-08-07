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

if [ ! -e "${MODEL_DIR}/models/object_detection/pytorch/ssd-resnet34/inference/cpu/infer.py" ]; then
  echo "Could not find the script of infer.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the infer.py exist at the: \${MODEL_DIR}/models/object_detection/pytorch/ssd-resnet34/inference/cpu/infer.py"
  exit 1
fi

if [ ! -e "${CHECKPOINT_DIR}/pretrained/resnet34-ssd1200.pth" ]; then
  echo "The pretrained model \${CHECKPOINT_DIR}/pretrained/resnet34-ssd1200.pth does not exist"
  exit 1
fi

if [ ! -d "${DATASET_DIR}/coco" ]; then
  echo "The DATASET_DIR \${DATASET_DIR}/coco does not exist"
  exit 1
fi

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "The OUTPUT_DIR '${OUTPUT_DIR}' does not exist"
  exit 1
fi

ARGS=""
if [ "$1" == "int8" ]; then
    ARGS="$ARGS --int8"
    ARGS="$ARGS --seed 1 --threshold 0.2 --configure $2"
    export DNNL_GRAPH_CONSTANT_CACHE=1
    echo "### running int8 datatype"
    ARGS="$ARGS --calibration"
    echo "### doing int8 calibration"
    ARGS="$ARGS --iter $3"
elif [ "$1" == "bf16" ]; then
    ARGS="$ARGS --autocast"
    echo "### int8 calibration not support bf16 datatype"
    exit -1
else
    echo "### int8 calibration not support running fp32 datatype"
    exit -1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export USE_IPEX=1
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

BATCH_SIZE=16

rm -rf ${OUTPUT_DIR}/accuracy_log*

python -m intel_extension_for_pytorch.cpu.launch \
    --memory-allocator jemalloc \
    ${MODEL_DIR}/models/object_detection/pytorch/ssd-resnet34/inference/cpu/infer.py \
    --data ${DATASET_DIR}/coco \
    --device 0 \
    --checkpoint ${CHECKPOINT_DIR}/pretrained/resnet34-ssd1200.pth \
    -j 0 \
    --no-cuda \
    --batch-size ${BATCH_SIZE} \
    --jit \
    $ARGS 2>&1 | tee ${OUTPUT_DIR}/accuracy_log.txt

# For the summary of results
wait

accuracy=$(grep 'Accuracy:' ${OUTPUT_DIR}/accuracy_log* |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
echo ""SSD-RN34";"accuracy";$1; ${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
