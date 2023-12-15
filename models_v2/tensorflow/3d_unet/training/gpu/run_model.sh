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
input_envs[PRECISION]=${PRECISION}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[DATASET_DIR]=${DATASET_DIR}
input_envs[MULTI_TILE]=${MULTI_TILE}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

if [ -d ${DATASET_DIR} ];then
  echo "DATASET_DIR is: "${DATASET_DIR}
else
  echo "Error: the path of dataset does not exist!"
  exit 1
fi

BATCH_SIZE=${BATCH_SIZE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD}

if [ ${PRECISION} == "bfloat16" ];then
  echo "Datatype is bfloat16"
  AMP="--amp"
elif [ ${PRECISION} == "fp32" ];then
  echo "Datatype is fp32"
  AMP=""
else
  echo "Error: "${DATATYPE}" not supported yet!"
  exit 1
fi

echo 'Running with parameters:'
echo " DATASET_PATH: ${DATASET_DIR}"
echo " OUTPUT_DIR: ${OUTPUT_DIR}"
echo " PRECISION: ${PRECISION}"
echo " BATCH_SIZE: $BATCH_SIZE"
echo " MULTI_TILE: $MULTI_TILE"

if [ $MULTI_TILE == "True" ];then
  cd 3d_unet_hvd/DeepLearningExamples/TensorFlow/Segmentation/UNet_3D_Medical/
  mpirun -np 2 -prepend-rank -ppn 2 \
  python main.py --data_dir $DATASET_DIR --benchmark --model_dir $OUTPUT_DIR \
  --exec_mode train --warmup_steps 150 --max_steps 1000 --batch_size $BATCH_SIZE \
  $AMP |& tee Unet3D_training_${PRECISION}_BS${BATCH_SIZE}_${MULTI_TILE}.log
  value=$(cat ./Unet3D_training_${PRECISION}_BS${BATCH_SIZE}_${MULTI_TILE}.log | grep -oE 'total_throughput_train : [0-9.]+' | awk '{print $NF}')
else
  cd 3d_unet/DeepLearningExamples/TensorFlow/Segmentation/UNet_3D_Medical/
  python main.py --benchmark --data_dir $DATASET_DIR --model_dir $OUTPUT_DIR \
  --exec_mode train --batch_size $BATCH_SIZE --warmup_steps 150 --max_steps 1000 --log_every 1 \
  $AMP |& tee Unet3D_training_${PRECISION}_BS${BATCH_SIZE}_${MULTI_TILE}.log
  value=$(cat ./Unet3D_training_${PRECISION}_BS${BATCH_SIZE}_${MULTI_TILE}.log | grep "Throughput is" | sed -e "s/.*is//")
fi

key="throughput"
unit="images/sec"

yaml_content=$(cat <<EOF
results:
 - key: $key
   value: $value
   unit: $unit
EOF
)

# Write the content to a YAML file
script_path="$(realpath "$0")"
script_directory=$(dirname "$script_path")
echo "$yaml_content" >  ${script_directory}/results.yaml
echo "YAML file created, path:$script_directory/results.yaml"