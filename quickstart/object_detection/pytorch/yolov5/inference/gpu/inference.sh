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
GPU_TYPE=${GPU_TYPE-flex_170}
PRECISION=${PRECISION-fp16}

if [[ -z "${IMAGE_FILE}" ]]; then
  echo "Please specify coco image file variable IMAGE_FILE"
  exit 1
fi

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [[ -z "${GPU_TYPE}" ]]; then
  echo "The required environment variable GPU_TYPE has not been set. Please specify either flex_170 or flex_140. "
  exit 1
fi

if [[ -z "${PRECISION}" ]]; then
  echo "The required environment variable$PRECISION has not been set. Please set it as fp16 "
  exit 1
fi
declare -a str
device_id=$( lspci | grep -i display | sed -n '1p' | awk '{print $7}' )
num_devs=$(lspci | grep -i display | awk '{print $7}' | wc -l)
num_threads=1
k=0

if [[ "$PRECISION" == "fp16" ]];then
  if [[ $GPU_TYPE == "flex_170" ]]; then 
    if [[ ${device_id} == "56c0" ]]; then 
      echo "YOLO(v5) Inference with FP16 BATCH SIZE ${BATCH_SIZE} on Flex 170"
      python -u ${MODEL_DIR}/models/object_detection/pytorch/yolov5/inference/gpu/detect.py \
      --source ${IMAGE_FILE} \
      --bs ${BATCH_SIZE} \
      --half \
      --iters ${NUM_ITERATIONS} \
      --dummy 1 \
      --benchmark 1 2>&1 | tee $OUTPUT_DIR/YOLOv5__xpu_inf_${BATCH_SIZE}.log
    fi
  elif [[ $GPU_TYPE == "flex_140" ]]; then
    if [[ ${device_id} == "56c1" ]]; then
      if [[ $BATCH_SIZE == 1 ]]; then
        echo "YOLO(v5) Inference with FP16 BATCH SIZE 1 on Flex 140"
        for i in $( eval echo {0..$((num_devs-1))} )
        do
          for j in $( eval echo {1..$num_threads} )
          do
              str+=("ZE_AFFINITY_MASK="${i}" numactl -C ${k} -l python -u ${MODEL_DIR}/models/object_detection/pytorch/yolov5/inference/gpu/detect.py \
            --source ${IMAGE_FILE} \
            --bs ${BATCH_SIZE} \
            --half \
            --iters ${NUM_ITERATIONS} \
            --dummy 1 \
            --benchmark 1 ")
            ((k=k+1))
          done
          done
          parallel --lb -d, --tagstring "[{#}]" ::: "${str[@]}" 2>&1 | tee $OUTPUT_DIR/YOLOv5__xpu_inf_c0_c1_${BATCH_SIZE}.log
      else
      echo "YOLO(v5) Inference with FP16 BATCH SIZE $BATCH_SIZE on Flex 140"
       for i in $( eval echo {0..$((num_devs-1))} )
        do
          str+=("ZE_AFFINITY_MASK="${i}" python -u ${MODEL_DIR}/models/object_detection/pytorch/yolov5/inference/gpu/detect.py \
            --source ${IMAGE_FILE} \
            --bs ${BATCH_SIZE} \
            --half \
            --iters ${NUM_ITERATIONS} \
            --dummy 1 \
            --benchmark 1 ")
        done
      parallel --lb -d, --tagstring "[{#}]" ::: "${str[@]}" 2>&1 | tee $OUTPUT_DIR/YOLOv5__xpu_inf_c0_c1_${BATCH_SIZE}.log
      fi
      file_loc=$OUTPUT_DIR/YOLOv5__xpu_inf_c0_c1_${BATCH_SIZE}.log
      total_throughput=$( cat $file_loc | grep Throughput | awk '{print $3}' | awk '{ sum_total += $1 } END { print sum_total }' )
      echo 'Total Throughput in images/sec: '$total_throughput | tee -a $file_loc
    fi
  fi
else 
  echo "Yolov5 currently supports FP16 precision"
  exit 1
fi
