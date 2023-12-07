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
input_envs[DATASET_DIR]=${DATASET_DIR}
input_envs[PRETRAINED_DIR]=${PRETRAINED_DIR}
input_envs[PRECISION]=${PRECISION}

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

BATCH_SIZE=${BATCH_SIZE:-16}

if [ ${PRECISION} == "fp16" ];then
  AMP=" --amp "
else
  AMP=""
fi

echo 'Running with parameters:'
echo " DATASET_PATH: ${DATASET_DIR}"
echo " PRETRAINED_DIR: ${PRETRAINED_DIR}"
echo " PRECISION: ${PRECISION}"
echo " BATCH_SIZE: $BATCH_SIZE"

cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN

python scripts/inference.py \
  --data_dir=$DATASET_DIR \
  --batch_size=$BATCH_SIZE \
  --no_xla \
  --weights_dir=$PRETRAINED_DIR $AMP |& tee maskrcnn_inference_${PRECISION}_BS${BATCH_SIZE}.log

throughput=$(cat maskrcnn_inference_${PRECISION}_BS${BATCH_SIZE}.log | grep Throughput | awk -F ' ' '{print $5}')
yaml_content=$(cat <<EOF
results:
 - key: throughput
   value: $throughput
   unit: records/sec
EOF
)

# Write the content to a YAML file
echo "$yaml_content" >  ./results.yaml
echo "YAML file created."
