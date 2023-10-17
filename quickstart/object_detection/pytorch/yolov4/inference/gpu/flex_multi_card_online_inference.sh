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
BATCH_SIZE=${BATCH_SIZE-1}
NUM_ITERATIONS=${NUM_ITERATIONS-5000}

if [[ -z "${PRETRAINED_MODEL}" ]]; then
  echo "The required environment variable PRETRAINED_MODEL has not been set."
  echo "Please specify a directory where the model weights were downloaded"
  exit 1
fi

export OverrideDefaultFP64Settings=1 
export IGC_EnableDPEmulation=1 
export CFESingleSliceDispatchCCSMode=1
export IPEX_ONEDNN_LAYOUT=1
export IPEX_LAYOUT_OPT=1
export IPEX_XPU_ONEDNN_LAYOUT=1

declare -a str
device_id=$( lspci | grep -i display | sed -n '1p' | awk '{print $7}' )
num_devs=$(lspci | grep -i display | awk '{print $7}' | wc -l)
num_threads=1
k=0

if [[ ${device_id} == "56c1" ]]; then
 for i in $( eval echo {0..$((num_devs-1))} )
    do
    for j in $( eval echo {1..$num_threads} )
    do
    str+=("ZE_AFFINITY_MASK="${i}" numactl -C ${k} -l  python -u ${MODEL_DIR}/models/object_detection/pytorch/yolov4/inference/gpu/models.py \
          -n 80 \
          -i ${IMAGE_FILE} \
          --weight ${PRETRAINED_MODEL} \
          -e 416 \
          -w 416 \
          -d int8 \
          --dummy 1 \
          -b ${BATCH_SIZE} \
          --benchmark 1 \
          --iter ${NUM_ITERATIONS} ")
    ((k=k+1))
    done
    done

echo "YOLOv4 dummy data int8 inference block nchw on Flex Series 140"
parallel --lb -d, --tagstring "[{#}]" ::: \
"${str[@]}" 2>&1 | tee $OUTPUT_DIR/YOLOv4_dummy_data_xpu_inf_c0_c1_${BATCH_SIZE}.log
file_loc=$OUTPUT_DIR/YOLOv4_dummy_data_xpu_inf_c0_c1_${BATCH_SIZE}.log
total_fps=$( cat $file_loc | grep FPS | awk '{print $4}' | awk '{ sum_total += $1 } END { print sum_total }' )
echo 'Total FPS: '$total_fps | tee -a $file_loc
fi
