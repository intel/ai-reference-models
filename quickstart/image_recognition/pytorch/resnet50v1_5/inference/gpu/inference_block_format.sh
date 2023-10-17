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
NUM_ITERATIONS=${NUM_ITERATIONS-500}
BATCH_SIZE=${BATCH_SIZE-1024}

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
declare -a str

export OverrideDefaultFP64Settings=1 
export IGC_EnableDPEmulation=1
export CFESingleSliceDispatchCCSMode=1
export IPEX_ONEDNN_LAYOUT=1
export IPEX_LAYOUT_OPT=1
export IPEX_XPU_ONEDNN_LAYOUT=1

if [[ -z "${NUM_OAM}" ]]; then
    # int8 uses a different python script
    echo "resnet50 int8 inference block"
    IPEX_XPU_ONEDNN_LAYOUT=1 python -u models/image_recognition/pytorch/resnet50v1_5/inference/gpu/main.py \
        -a resnet50 \
        -b ${BATCH_SIZE} \
        --xpu 0 \
        -e \
        --pretrained \
        --int8 1 \
        --num-iterations ${NUM_ITERATIONS} \
        --benchmark 1 \
        ${dataset_arg}  2>&1 | tee ${OUTPUT_DIR}//resnet50_int8_inf_block_t0_raw.log

elif [[ ${NUM_OAM} == "4" ]]; then
    echo "resnet50 int8 inference block on ${NUM_OAM} OAM module(s)"
    NUM_TILES_PER_GPU=2
    for i in $( eval echo {0..$((NUM_OAM-1))} )
        do
            for j in $( eval echo {0..$((NUM_TILES_PER_GPU-1))} )
                do
                    str+=("ZE_AFFINITY_MASK="${i}"."${j}" IPEX_XPU_ONEDNN_LAYOUT=1 python -u models/image_recognition/pytorch/resnet50v1_5/inference/gpu/main.py \
                            -a resnet50 \
                            -b ${BATCH_SIZE} \
                            --xpu 0 \
                            -e \
                            --pretrained \
                            --int8 1 \
                            --num-iterations ${NUM_ITERATIONS} \
                            --benchmark 1 \
                            ${dataset_arg}  2>&1 | tee ${OUTPUT_DIR}/resnet50_int8_inf_block_c${i}_t${j}_raw.log & ")
                done
        done
    str=${str[@]}
    cmd_line=${str::-2}
    eval $cmd_line
else
    echo "Currently only x4 OAM Modules are supported"
    exit 1
fi
