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
if [ ! -d "${DATASET_DIR}/coco" ]; then
  echo "The DATASET_DIR \${DATASET_DIR}/coco does not exist"
  exit 1
fi

if [ ! -e "${CHECKPOINT_DIR}/yolov7.pt" ]; then
  echo "The CHECKPOINT_DIR \${CHECKPOINT_DIR}/yolov7.pt does not exist"
  exit 1
fi

cd $DATASET_DIR
DATASET_DIR=$(pwd)
cd -

cd $CHECKPOINT_DIR
CHECKPOINT_DIR=$(pwd)
cd -

MODEL_DIR=${MODEL_DIR-$PWD}
if [ ! -e "${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7_ipex.patch"  ]; then
    echo "Could not find the script of yolov7_ipex.patch. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the yolov7_ipex.patch exist at the: \${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7_ipex.patch"
    exit 1
else
    TMP_PATH=$(pwd)
    cd "${MODEL_DIR}/models/object_detection/pytorch/yolov7/"
    if [ ! -d "yolov7" ]; then
        git clone https://github.com/WongKinYiu/yolov7.git yolov7
        cd yolov7
        cp ../yolov7_int8_default_qparams.json .
        cp ../yolov7.py .
        pip install -r requirements.txt
        git checkout a207844
        git apply ../yolov7_ipex.patch
    else
        cd yolov7
    fi
    cd $TMP_PATH
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

cd ${OUTPUT_DIR}
OUTPUT_DIR=$(pwd)
cd -


export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

BATCH_SIZE=${BATCH_SIZE:-40}

rm -rf ${OUTPUT_DIR}/yolov7_int8_calibration.log

cd "${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7"

ARGS="--checkpoint-dir $CHECKPOINT_DIR --weights yolov7.pt"
ARGS="$ARGS --img 640 -e --data data/coco.yaml --dataset-dir $DATASET_DIR --conf-thres 0.001 --iou 0.65 --device cpu"

echo "running int8 calibration"

ARGS="$ARGS --int8 --calibration --configure-dir $1 --calibration-steps $2"

python -m intel_extension_for_pytorch.cpu.launch \
    --memory-allocator default \
    --ninstances 1 \
    ${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7/yolov7.py \
    $ARGS \
    --ipex \
    --batch-size $BATCH_SIZE 2>&1 | tee ${OUTPUT_DIR}/yolov7_int8_calibration.log

wait
cd -

echo "calibrated file is save to ${MODEL_DIR}/models/object_detection/pytorch/yolov7/yolov7/${1}"