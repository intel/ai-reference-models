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
# SPDX-License-Identifier: EPL-2.0
#

#!/bin/bash

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[PRECISION]=${PRECISION}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
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

BATCH_SIZE=${BATCH_SIZE:-4}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD}
EPOCHS=${EPOCHS:-1}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-20}

if [ ${PRECISION} == "bfloat16" ];then
  echo "PRECISION is bfloat16"
  AMP="--amp"
else
  echo "PRECISION is "${PRECISION}
  AMP=""
fi

echo 'Running with parameters:'
echo " DATASET_PATH: ${DATASET_DIR}"
echo " OUTPUT_DIR: ${OUTPUT_DIR}"
echo " PRECISION: ${PRECISION}"
echo " BATCH_SIZE: $BATCH_SIZE"
echo " EPOCHS: $EPOCHS"
echo " STEPS_PER_EPOCH: $STEPS_PER_EPOCH" 

cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN

python main.py train \
--data_dir $DATASET_DIR \
--model_dir=$OUTPUT_DIR \
--train_batch_size $BATCH_SIZE \
--seed=0 --use_synthetic_data \
--epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH \
--log_every=1 --log_warmup_steps=1 \
$AMP

