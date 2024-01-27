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
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[GPU_TYPE]=${GPU_TYPE}
input_envs[PRECISION]=${PRECISION}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}

  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

if [ -d "${DATASET_DIR}" ]; then
  echo "DATASET_DIR is: "${DATASET_DIR}
else
  echo "Error: the path of dataset does not exist!"
  exit 1
fi

BATCH_SIZE=${BATCH_SIZE:-16}

if [ "${PRECISION}" == "float16" ]; then
  echo "PRECISION is float16"
  AMP="--amp"
else
  AMP=""
  echo "Only float16 PRECISION is supported."
  exit 1
fi

echo 'Running with parameters:'
echo " DATASET_PATH: ${DATASET_DIR}"
echo " PRECISION: ${PRECISION}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo " BATCH_SIZE: ${BATCH_SIZE}"

mkdir -p PRETRAINED_WEIGHTS
python -u ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/scripts/download_weights.py --save_dir=$PRETRAINED_WEIGHTS

declare -a str
device_id=$( lspci | grep -i display | sed -n '1p' | awk '{print $7}' )
num_devs=$(lspci | grep -i display | awk '{print $7}' | wc -l)
num_threads=1
k=0

cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN

 if [[ ${GPU_TYPE} == flex_170 ]]; then
    if [[ ${device_id} == "56c0" ]]; then
      echo "Running ${PRECISION} MaskRCNN Inference on Flex 170"
      python scripts/inference.py \
      --data_dir=$DATASET_DIR \
      --batch_size=$BATCH_SIZE \
      --no_xla \
      --weights_dir=$PRETRAINED_WEIGHTS $AMP |& tee ${OUTPUT_DIR}/Maskrcnn_inference_${PRECISION}.log
      value=$(cat ${OUTPUT_DIR}/Maskrcnn_inference_${PRECISION}.log | grep -o "'predict_throughput': [0-9.]*" | awk -F ": " '{print $2}' | tail -1)
    fi
elif [[ ${GPU_TYPE} == flex_140 ]]; then
    if [[ ${device_id} == "56c1" ]]; then
      BATCH_SIZE=1
      echo "Running ${PRECISION} MaskRCNN Inference with BATCH SIZE 1 on Flex 140"
      for i in $( eval echo {0..$((num_devs-1))} )
          do
            for j in $( eval echo {1..$num_threads} )
            do
            str+=("ZE_AFFINITY_MASK="${i}" numactl -C ${k} -l \
            python scripts/inference.py \
            --data_dir=$DATASET_DIR \
            --batch_size=$BATCH_SIZE \
            --no_xla \
            --weights_dir=$PRETRAINED_WEIGHTS $AMP ")
            ((k=k+1))
            done
          done
      parallel --lb -d, --tagstring "[{#}]" ::: "${str[@]}" 2>&1 | tee ${OUTPUT_DIR}/Maskrcnn_inference_${PRECISION}.log
      value=$(grep -rnw "5000/Unknown" ${OUTPUT_DIR}/Maskrcnn_inference_${PRECISION}.log | grep -o "'predict_throughput': [0-9.]*" | awk -F ": " '{print $2}' | tail -2 | awk '{ sum_total += $1 } END { print sum_total }')
    fi
fi
key="throughput"
unit="images/sec"

throughput=$(cat maskrcnn_inference_${PRECISION}_BS${BATCH_SIZE}.log | grep Throughput | awk -F ' ' '{print $5}')
yaml_content=$(cat <<EOF
results:
 - key: throughput
   value: $throughput
   unit: records/sec
EOF
)

# Write the content to a YAML file
echo "$yaml_content" >  ${OUTPUT_DIR}/results.yaml
echo "YAML file created, path:${OUTPUT_DIR}/results.yaml"
