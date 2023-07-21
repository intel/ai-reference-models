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
BATCH_SIZE=${BATCH_SIZE-1024}

echo 'MODEL_DIR='$MODEL_DIR
echo 'PRECISION='$PRECISION
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

if [[ ! -f "${FROZEN_GRAPH}" ]]; then
  pretrained_model=/workspace/tf-flex-series-ssd-mobilenet-inference/pretrained_models/ssdmobilenet_${PRECISION}_pretrained_model_gpu.pb
else
  pretrained_model=${FROZEN_GRAPH}
fi

export TF_NUM_INTEROP_THREADS=1
export OverrideDefaultFP64Settings=1 
export IGC_EnableDPEmulation=1 
export CFESingleSliceDispatchCCSMode=1


# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[PRECISION]=${PRECISION}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}

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

WARMUP=""
if [[ $PRECISION == "int8" ]]; then
  WARMUP="-- warmup_steps=5 steps=5000"
  else
  echo "Flex series GPU SUPPORTS ONLY INT8 PRECISION"
  exit 1
fi
declare -a str
device_id=$( lspci | grep -i display | sed -n '1p' | awk '{print $7}' )
num_devs=$(lspci | grep -i display | awk '{print $7}' | wc -l)
source "${MODEL_DIR}/quickstart/common/utils.sh"

mac_0=`sudo lspci | grep Dis| head -n 1| awk '{print $1}'`
node_0=`sudo lspci -s $mac_0 -v | grep NUMA | awk -F, '{print $5}' | awk '{print $3}'`
mac_1=`sudo lspci | grep Dis| tail -n 1| awk '{print $1}'`
node_1=`sudo lspci -s $mac_1 -v | grep NUMA | awk -F, '{print $5}' | awk '{print $3}'`

j=$node_0
if [[ ${device_id} == "56c1" ]]; then
    for i in $( eval echo {0..$((num_devs-1))} )
    do
    str+=("ZE_AFFINITY_MASK="${i}" numactl -N "${j}" -l python -u models/object_detection/tensorflow/ssd-mobilenet/inference/gpu/int8/infer_detections.py \
      --input-graph ${pretrained_model} \
      --batch-size ${BATCH_SIZE} \
      --iter 5000 \
      --warmup_iter 5 \
      --benchmark ")
    j=$node_1
    done
    # int8 uses a different python script
    echo "ssd-mobilenet int8 inference block on Flex series 140"
    parallel --lb -d, --tagstring "[{#}]" ::: \
    "${str[@]}" 2>&1 | tee ${OUTPUT_DIR}//ssd-mobilenet_${PRECISION}_inf_c0_c1_${BATCH_SIZE}.log
    file_loc=${OUTPUT_DIR}//ssd-mobilenet_${PRECISION}_inf_c0_c1_${BATCH_SIZE}.log
    total_fps=$( cat $file_loc | grep 'Total samples/sec' | awk '{print $4}' | awk '{ sum_total += $1 } END { print sum_total }' )
    echo 'Total FPS: '$total_fps | tee -a $file_loc
fi
