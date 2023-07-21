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
NUM_ITERATIONS=${NUM_ITERATIONS:-5000}
echo 'label='$label

if [[ -z "${DATASET_DIR}" ]]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

# Create the model weights directory, if it doesn't already exist
mkdir -p ${MODEL_DIR}/PRETRAINED_MODEL

export OverrideDefaultFP64Settings=1 
export IGC_EnableDPEmulation=1 

export CFESingleSliceDispatchCCSMode=1
export IPEX_ONEDNN_LAYOUT=1
export IPEX_LAYOUT_OPT=1
export IPEX_XPU_ONEDNN_LAYOUT=1

# Download the weights file if it does not already exist
WEIGHTS_FILE="${MODEL_DIR}/PRETRAINED_MODEL/mobilenet-v1-ssd-mp-0_675.pth"
wget https://drive.google.com/uc?id=1pSPLnWGGNs3kV_YSxr4vsmSvDCLpUsEr -O $WEIGHTS_FILE

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs

input_envs[label]=${label}

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

calculate_throughput() {
  file_loc=$1
  batch_size=$2
  itr=$( cat $file_loc | awk '{print $1}' | tr -d "[]" | sort -n | tail -n1 )
  for i in $( eval echo {1..$itr} )
  do
  fps=$(grep -rnw [$i] $file_loc | grep 'Inference time' | tail -4900 | awk -v batch_size="$batch_size" -F' ' '{sum+=$NF;} END{print batch_size/(sum/4900)} ')
  echo 'FPS: '$fps | tee -a $file_loc
  done
  total_fps=$(cat $file_loc | grep FPS | awk '{print $2}' | awk '{ sum_total += $1 } END { print sum_total }' )
  echo 'Total FPS: '$total_fps | tee -a $file_loc
}
# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

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
            str+=("ZE_AFFINITY_MASK="${i}" numactl -C ${k} -l  python -u ${MODEL_DIR}/models/object_detection/pytorch/ssd-mobilenet/inference/gpu/eval_ssd.py \
                  --net mb1-ssd \
                  --dataset ${DATASET_DIR} \
                  --trained_model ${WEIGHTS_FILE} \
                  --label_file ${label} \
                  --dummy 1 \
                  --batch_size ${BATCH_SIZE} \
                  --benchmark 1 \
                  --num-iterations ${NUM_ITERATIONS} \
                  --int8 ")
    ((k=k+1))
    done
    done
echo "ssd-mobilenet dummy data inference  nchw on Flex Series 140"
parallel --lb -d, --tagstring "[{#}]" ::: \
"${str[@]}" 2>&1 | tee $OUTPUT_DIR/ssd_mobilenetv1_dummy_data_xpu_inf_c0_c1_${BATCH_SIZE}.log
calculate_throughput $OUTPUT_DIR/ssd_mobilenetv1_dummy_data_xpu_inf_c0_c1_${BATCH_SIZE}.log ${BATCH_SIZE}
fi
