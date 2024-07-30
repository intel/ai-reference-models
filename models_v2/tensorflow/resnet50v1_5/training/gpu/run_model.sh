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
input_envs[CONFIG_FILE]=${CONFIG_FILE}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
#input_envs[DATASET_DIR]=${DATASET_DIR}
input_envs[MULTI_TILE]=${MULTI_TILE}
input_envs[NUM_DEVICES]=${NUM_DEVICES}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

#BATCH_SIZE=${BATCH_SIZE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD}
script_path="$(realpath "$0")"
script_directory=$(dirname "$script_path")

echo 'Running with parameters:'
echo " DATASET_PATH: ${DATASET_DIR}"
echo " OUTPUT_DIR: ${OUTPUT_DIR}"
echo " CONFIG_FILE: ${CONFIG_FILE}"
echo " MULTI_TILE: ${MULTI_TILE}"
echo " NUM_DEVICES: ${NUM_DEVICES}"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
else
    rm -rf $OUTPUT_DIR && mkdir -p $OUTPUT_DIR                         
fi

export TF_NUM_INTEROP_THREADS=1  #for better performance
if [[ ${NUM_DEVICES} > 1 ]];then
  current_dir=$(pwd)
  if [ -d "tensorflow-models" ]; then
    echo "Repository already exists. Skipping clone."
  else
    mkdir $current_dir/resnet50_hvd/ && cd $current_dir/resnet50_hvd/
    git clone -b v2.14.0 https://github.com/tensorflow/models.git tensorflow-models
    cd tensorflow-models
    git apply $current_dir/hvd_support.patch
  fi
  export PYTHONPATH=$script_directory/resnet50_hvd/tensorflow-models
  mpirun -np ${NUM_DEVICES} -prepend-rank -ppn ${NUM_DEVICES} \
  python ${PYTHONPATH}/official/legacy/image_classification/classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=resnet \
  --dataset=imagenet \
  --model_dir=$OUTPUT_DIR \
  --data_dir=$DATASET_DIR \
  --config_file=$CONFIG_FILE |& tee Resnet50_training_MultiTile-${MULTI_TILE}_${NUM_DEVICES}devices.log
  value=0
  for i in $(seq 0 $((NUM_DEVICES-1)));do
    values[i]=$(cat Resnet50_training_MultiTile-${MULTI_TILE}_${NUM_DEVICES}devices.log | grep examples/second | grep "\[$i\]" | tail -1 | awk -F 'examples/second' '{print $1}' | awk -F ',' '{print $2}')
    value=$(echo "$value + ${values[i]}" | bc)
  done
else
  current_dir=$(pwd)
  if [ -d "tensorflow-models" ]; then
    echo "Repository already exists. Skipping clone."
  else
    mkdir $current_dir/resnet50/ && cd $current_dir/resnet50/
    git clone -b v2.14.0 https://github.com/tensorflow/models.git tensorflow-models
    cd tensorflow-models
    git apply $current_dir/resnet50.patch
    cd $current_dir
  fi
  export PYTHONPATH=$script_directory/resnet50/tensorflow-models
  python ${PYTHONPATH}/official/legacy/image_classification/classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=resnet \
  --dataset=imagenet \
  --model_dir=$OUTPUT_DIR \
  --data_dir=$DATASET_DIR \
  --config_file=$CONFIG_FILE |& tee Resnet50_training_MultiTile-${MULTI_TILE}_1device.log
  value=$(cat ./Resnet50_training_MultiTile-${MULTI_TILE}_1device.log | grep 'examples/second' | tail -1 | awk -F 'examples/second' '{print $1}' | awk -F ',' '{print $2}')
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

echo "$yaml_content" > $OUTPUT_DIR/results.yaml
echo "YAML file created, path: $OUTPUT_DIR/results.yaml"
