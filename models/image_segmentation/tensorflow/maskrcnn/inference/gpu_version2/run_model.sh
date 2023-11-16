#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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
# SPDX-License-Identifier: EPL-2.0
#

#!/bin/bash

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[PRECISION]=${PRECISION}
input_envs[DATASET_DIR]=${DATASET_DIR}

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

if [ -d ${PRETRAINED_DIR} ];then
  echo "PRETRAINED_DIR is: "${PRETRAINED_DIR}
else
  echo "Error: the path of pretrained does not exist!"
  exit 1
fi

BATCH_SIZE=${BATCH_SIZE:-16}


if [ ${PRECISION} == "float16" ];then
  echo "PRECISION is float16"
  AMP="--amp"
else
  echo "PRECISION is default fp32"
  AMP=""
fi

echo 'Running with parameters:'
echo " DATASET_PATH: ${DATASET_DIR}"
echo " PRECISION: ${PRECISION}"
echo " BATCH_SIZE: $BATCH_SIZE"
echo "PRETRAINED_DIR: $PRETRAINED_DIR"

cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN

python scripts/inference.py \
  --data_dir=$DATASET_DIR \
  --batch_size=$BATCH_SIZE \
  --no_xla \
  --weights_dir=$PRETRAINED_DIR $AMP |& tee Maskrcnn_inference_${PRECISION}.log


value=$(cat ./Maskrcnn_inference_${PRECISION}.log | grep -o "'predict_throughput': [0-9.]*" | awk -F ": " '{print $2}' | tail -1)
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