#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
input_envs[PRECISION]=${PRECISION}
input_envs[MODEL_PATH]=${MODEL_PATH}
input_envs[KERAS_BACKEND]=${KERAS_BACKEND}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

MAX_LENGTH=${MAX_LENGTH:-64}
BATCH_SIZE=${BATCH_SIZE:-128}

mkdir -p ${OUTPUT_DIR}

echo 'Running with parameters:'
echo " PRECISION: ${PRECISION}"
echo " MODEL_PATH: ${MODEL_PATH}"
echo " OUTPUT_DIR: ${OUTPUT_DIR}"
echo " MAX_LENGTH: ${MAX_LENGTH}"
echo " KERAS_BACKEND: ${KERAS_BACKEND}"
echo " BATCH_SIZE: ${BATCH_SIZE}"

RUN_CMD="python generate.py --precision=${PRECISION} --model_path=${MODEL_PATH} --max_length=${MAX_LENGTH} --keras_backend=${KERAS_BACKEND} --batch_size=${BATCH_SIZE}"

CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
CORES_PER_INSTANCE=`expr $CORES_PER_SOCKET \* $SOCKETS / $NUMAS`
echo "System configuration:"
echo "CORES_PER_SOCKET: $CORES_PER_SOCKET"
echo "SOCKETS: $SOCKETS"
echo "NUMAS: $NUMAS"
echo "CORES_PER_INSTANCE: $CORES_PER_INSTANCE"

echo "Running $NUMAS instances"
for (( i=0 ; i<$NUMAS ; i++ )); 
do
  echo "numactl --localalloc -N $i ${RUN_CMD} > ${OUTPUT_DIR}/gemma_inference_instance${i}.log 2>&1 &"
  numactl --localalloc -N $i ${RUN_CMD} > "${OUTPUT_DIR}/gemma_inference_instance${i}.log" 2>&1 &
done

wait # Wait for all background processes to complete

total_throughput=$(grep "Throughput:" "${OUTPUT_DIR}"/gemma_inference_instance*.log | awk '{sum += $2} END {print sum}')
echo "Total throughput: $total_throughput inputs/sec"

yaml_content=$(cat <<EOF
results:
 - key: total throughput
   value: $total_throughput
   unit: inputs/sec
EOF
)

# Write the content to a YAML file
echo "$yaml_content" >  ${OUTPUT_DIR}/results.yaml
echo "YAML file created."
