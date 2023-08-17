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
BATCH_SIZE=${BATCH_SIZE-32}
NUM_ITERATIONS=${NUM_ITERATIONS-500}

if [[ -z "${IMAGE_FILE}" ]]; then
  echo "Please specify coco image file variable IMAGE_FILE"
  exit 1
fi

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

echo "YOLO(v5) Inference FP16 BATCH SIZE ${BATCH_SIZE} Inference"
python -u ${MODEL_DIR}/models/object_detection/pytorch/yolov5/inference/gpu/detect.py \
  --source ${IMAGE_FILE} \
  --bs ${BATCH_SIZE} \
  --half \
  --iters ${NUM_ITERATIONS} \
  --dummy 1 \
  --benchmark 1 2>&1 | tee $OUTPUT_DIR/YOLOv5_FP16_bs${BATCH_SIZE}_inf.log
