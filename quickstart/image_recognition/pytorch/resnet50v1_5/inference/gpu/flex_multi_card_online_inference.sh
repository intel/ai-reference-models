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
NUM_ITERATIONS=${NUM_ITERATIONS-5000}
BATCH_SIZE=${BATCH_SIZE-1}

dataset_arg="${DATASET_DIR}"
if [[ -z "${DATASET_DIR}" ]]; then
  echo "Using Dummy data since environment variable DATASET_DIR has not been set"
  dataset_arg="--dummy"
elif [[ ! -d "${DATASET_DIR}" ]]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

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
            str+=("ZE_AFFINITY_MASK="${i}" numactl -C ${k} -l python -u models/image_recognition/pytorch/resnet50v1_5/inference/gpu/main.py -a resnet50 -b ${BATCH_SIZE} --xpu 0 -e --pretrained --int8 1 --num-iterations ${NUM_ITERATIONS} --benchmark 1 ${dataset_arg} ")
    ((k=k+1))
    done
    done
    # int8 uses a different python script
    echo "resnet50 int8 inference block on Flex series 140"
    parallel --lb -d, --tagstring "[{#}]" ::: \
    "${str[@]}" 2>&1 | tee ${OUTPUT_DIR}//resnet50_${PRECISION}_inf_block_c0_c1_${BATCH_SIZE}.log
    file_loc=${OUTPUT_DIR}//resnet50_${PRECISION}_inf_block_c0_c1_${BATCH_SIZE}.log
    total_throughput=$( cat $file_loc | grep throughput | awk '{print $7}' | cut -d':' -f2 | awk '{ sum_total += $1 } END { print sum_total }' )
    echo 'Total Throughput in images/sec: '$total_throughput | tee -a $file_loc
fi
