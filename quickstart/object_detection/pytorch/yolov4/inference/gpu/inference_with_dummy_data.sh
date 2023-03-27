#!/usr/bin/env bash
#
# Copyright (c) 2022 Intel Corporation
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

./quickstart/setup.sh

if [[ -z "${PRETRAINED_MODEL}" ]]; then
  echo "The required environment variable PRETRAINED_MODEL has not been set."
  echo "Please specify a directory where the model weights were downloaded"
  exit 1
fi

echo "YOLOv4 dummy data int8 inference block nchw"
IPEX_XPU_ONEDNN_LAYOUT=1 python -u ${MODEL_DIR}/models/object_detection/pytorch/yolov4/inference/gpu/models.py \
  -n 80 \
  --weight ${PRETRAINED_MODEL} \
  -e 416 \
  -w 416 \
  -name ${MODEL_DIR}/models/object_detection/pytorch/yolov4/inference/gpu/data/coco.names \
  -d int8 \
  --dummy 1 \
  -b 64 \
  --benchmark 1 \
  --iter 500 2>&1 | tee $OUTPUT_DIR/YOLOv4_dummy_data_xpu_inf.log
