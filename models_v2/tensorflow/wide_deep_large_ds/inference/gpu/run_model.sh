#
# -*- coding: utf-8 -*-
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
#

#!/bin/bash

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[DATASET_PATH]=${DATASET_PATH}
input_envs[PB_FILE_PATH]=${PB_FILE_PATH}
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

BATCH_SIZE=${BATCH_SIZE:-10000}

echo 'Running with parameters:'
echo " DATASET_PATH: ${DATASET_PATH}"
echo " PB_FILE_PATH: ${PB_FILE_PATH}"
echo " BATCH_SIZE: ${BATCH_SIZE}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo " BATCH_SIZE: ${BATCH_SIZE}"

declare -a str
device_id=$( lspci | grep -i display | sed -n '1p' | awk '{print $7}' )
num_devs=$(lspci | grep -i display | awk '{print $7}' | wc -l)

if [[ ${GPU_TYPE} == flex_170 ]]; then 
    if [[ ${device_id} == "56c0" ]]; then
      echo "Running FP16 Wide and Deep Large Inference on Flex 170"
      python inference.py  --input_graph=$PB_FILE_PATH \
             --data_location=$DATASET_PATH --batch_size=$BATCH_SIZE \
            --num_inter_threads=1 |& tee wide-deep-large-ds_inference_BS${BATCH_SIZE}.log
      throughput=$(cat wide-deep-large-ds_inference_BS${BATCH_SIZE}.log | grep Throughput | awk -F ' ' '{print $12}')
    fi
elif [[ ${GPU_TYPE} == flex_140 ]]; then 
    if [[ ${device_id} == "56c1" ]]; then
      echo "Running FP16 Wide and Deep Large Inference on Flex 140"
      for i in $( eval echo {0..$((num_devs-1))} )
        do
          str+=("ZE_AFFINITY_MASK="${i}" python inference.py  --input_graph=$PB_FILE_PATH \
             --data_location=$DATASET_PATH --batch_size=$BATCH_SIZE \
            --num_inter_threads=1 ")
        done
      parallel --lb -d, --tagstring "[{#}]" ::: "${str[@]}" |& tee wide-deep-large-ds_inference_BS${BATCH_SIZE}.log
      throughput=$(cat wide-deep-large-ds_inference_BS${BATCH_SIZE}.log | grep Throughput | awk -F ' ' '{print $13}' | awk '{ sum_total += $1 } END { print sum_total }' )
    fi
fi
 

yaml_content=$(cat <<EOF
results:
 - key: throughput
   value: $throughput
   unit: records/sec
EOF
)

# Write the content to a YAML file
echo "$yaml_content" >  ${OUTPUT_DIR}/results.yaml
echo "YAML file created."
