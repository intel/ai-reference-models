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

echo 'MODEL_DIR='$MODEL_DIR
echo 'PRECISION='$PRECISION
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

export TF_NUM_INTEROP_THREADS=1
export CFESingleSliceDispatchCCSMode=1
export ITEX_LIMIT_MEMORY_SIZE_IN_MB=1024

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[PRECISION]=${PRECISION}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[GPU_TYPE]=${GPU_TYPE}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

export OverrideDefaultFP64Settings=1 
export IGC_EnableDPEmulation=1 
export TF_NUM_INTEROP_THREADS=1
export CFESingleSliceDispatchCCSMode=1
export ITEX_LIMIT_MEMORY_SIZE_IN_MB=1024

if [[ $PRECISION == "int8" ]]; then
    echo "Precision is $PRECISION"
    if [[ ! -f "${FROZEN_GRAPH}" ]]; then
      pretrained_model=/workspace/tf-flex-series-resnet50v1-5-inference/pretrained_models/resnet50v1_5-frozen_graph-${PRECISION}-gpu.pb
    else
      pretrained_model=${FROZEN_GRAPH}
    fi
    # WARMUP="-- warmup_steps=10 steps=5000"
  else 
    echo "FLEX SERIES GPU SUPPORTS ONLY INT8 PRECISION"
    exit 1
fi

declare -a str
device_id=$( lspci | grep -i display | sed -n '1p' | awk '{print $7}' )
num_devs=$(lspci | grep -i display | awk '{print $7}' | wc -l)
num_threads=1
k=0
# source "${MODEL_DIR}/quickstart/common/utils.sh"
if [[ ${device_id} == "56c1" ]]; then
    for i in $( eval echo {0..$((num_devs-1))} )
    do
    for j in $( eval echo {1..$num_threads} )
    do
    str+=("ZE_AFFINITY_MASK="${i}" numactl -C ${k} -l python -u models/image_recognition/tensorflow/resnet50v1_5/inference/gpu/int8/eval_image_classifier_inference.py \
          --input-graph ${pretrained_model} \
          --warmup-steps 10 \
          --steps 5000 \
          --batch-size ${BATCH_SIZE} \
          --benchmark ")
    ((k=k+1))
    done
    done
    # int8 uses a different python script
    echo "resnet50 int8 inference block on Flex series 140"
    parallel --lb -d, --tagstring "[{#}]" ::: \
    "${str[@]}" 2>&1 | tee ${OUTPUT_DIR}//resnet50_${PRECISION}_inf_block_c0_c1_${BATCH_SIZE}.log
    file_loc=${OUTPUT_DIR}//resnet50_${PRECISION}_inf_block_c0_c1_${BATCH_SIZE}.log
    total_throughput=$( cat $file_loc | grep Throughput | awk '{print $3}' |  awk '{ sum_total += $1 } END { print sum_total }' )
    echo 'Total Throughput in images/sec: '$total_throughput | tee -a $file_loc
fi
