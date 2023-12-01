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
input_envs[PB_FILE_PATH]=${PB_FILE_PATH}
#input_envs[DATASET_DIR]=${DATASET_DIR} 
#input_envs[BATCH_SIZE]=${BATCH_SIZE} #if not set batch size, the default 1024 will use
input_envs[TEST_MODE]=${TEST_MODE}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

#dataset only need for accuracy
if [ -n "${DATASET_DIR}" ];then
  if [ -d ${DATASET_DIR} ];then
    echo "DATASET_DIR is "${DATASET_DIR}
  else
    echo "Error: the path of dataset does not exist!"
    exit 1
  fi
elif [ ${TEST_MODE} == "accuracy" ];then
  echo "Error: the path of dataset is required!"
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-1024}"
echo 'Running with parameters:'
echo " DATASET_PATH: ${DATASET_DIR}"
echo " PB_FILE_PATH: ${PB_FILE_PATH}"
echo " PRECISION: ${PRECISION}"
echo " BATCH_SIZE: ${BATCH_SIZE}"
echo " MODE: ${TEST_MODE}" 

if [ ${PRECISION} == "float16" ];then
  export ITEX_AUTO_MIXED_PRECISION=1
  export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="FLOAT16"
elif [ ${PRECISION} == "bfloat16" ];then
  export ITEX_AUTO_MIXED_PRECISION=1
  export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16"
elif [ ${PRECISION} == "tensorfloat32" ];then
  export ITEX_FP32_MATH_MODE=TF32
elif [ ${PRECISION} == "float32" ];then
  echo "Using default datatype: float32"
elif [ ${PRECISION} == "int8" ];then
  ARGS="--benchmark"
  echo "Using default datatype: int8"
else
  echo "Error: Only support float32/bfloat16/float16/tensorfloat32/int8"
  exit 1    
fi

if [ ${TEST_MODE} == "accuracy" ];then
  python eval_image_classifier_inference.py --input-graph=${PB_FILE_PATH} --batch-size=${BATCH_SIZE} --data-num-inter-threads 1 --accuracy-only --data-location ${DATASET_DIR} --dtype ${PRECISION} \
  |& tee resnet50_${TEST_MODE}_${PRECISION}.log
elif [ ${TEST_MODE} == "inference" ];then
  echo "Running benchmark"
  python eval_image_classifier_inference.py --input-graph=${PB_FILE_PATH} --batch-size=${BATCH_SIZE} --warmup-steps=10 --steps=5000 --dtype ${PRECISION} ${ARGS} \
  |& tee resnet50_${TEST_MODE}_${PRECISION}.log
fi

if [ ${TEST_MODE} == "accuracy" ];then
  value=$(cat resnet50_${TEST_MODE}_${PRECISION}.log | grep "(Top1 accuracy, Top5 accuracy) " | tail -n 1 | sed -e "s/.*(//" | sed -e "s/,.*//")
  key="accuracy"
  unit=""
elif [ ${TEST_MODE} == "inference" ];then 
  value=$(cat resnet50_${TEST_MODE}_${PRECISION}.log | grep "Throughput" | sed -e "s/.*://" | sed -e "s/ images\/sec//")
  key="throughput"
  unit="images/s"
fi



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
