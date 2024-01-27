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
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

mkdir -p ${OUTPUT_DIR}

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

declare -a str
device_id=$( lspci | grep -i display | sed -n '1p' | awk '{print $7}' )
num_devs=$(lspci | grep -i display | awk '{print $7}' | wc -l)
num_threads=1
k=0

if [ ${TEST_MODE} == "accuracy" ];then
  python eval_image_classifier_inference.py --input-graph=${PB_FILE_PATH} --batch-size=${BATCH_SIZE} --data-num-inter-threads 1 --accuracy-only --data-location ${DATASET_DIR} --dtype ${PRECISION} \
  |& tee resnet50_${TEST_MODE}_${PRECISION}.log
elif [ ${TEST_MODE} == "inference" ];then
    if [[ -z ${FLEX_GPU_TYPE} ]]; then
      echo "FLEX_GPU_TYPE not set. Please set either flex_170 or flex_140"
      exit 0
    fi  
  if [[ ${FLEX_GPU_TYPE} == flex_140 ]]; then 
    if [[ ${device_id} == "56c1" ]]; then
  	echo "Running benchmark"
	if [[ ${BATCH_SIZE} == "1" ]]; then
	  for i in $( eval echo {0..$((num_devs-1))} )
    	do
   	  for j in $( eval echo {1..$num_threads} )
           do
		   str+=("ZE_AFFINITY_MASK="${i}" numactl -C ${k} -l python eval_image_classifier_inference.py --input-graph=${PB_FILE_PATH} --batch-size=${BATCH_SIZE} --warmup-steps=10 --steps=5000 --dtype ${PRECISION} ${ARGS} ")
		   ((k=k+1))
	    done
    done
  else
	    for i in $( eval echo {0..$((num_devs-1))} )
	    do
		    str+=("ZE_AFFINITY_MASK="${i}" python eval_image_classifier_inference.py --input-graph=${PB_FILE_PATH} --batch-size=${BATCH_SIZE} --warmup-steps=10 --steps=5000 --dtype ${PRECISION} ${ARGS} ")
	    done
	fi
echo "resnet50 int8 inference on Flex series 140"
parallel --lb -d, --tagstring "[{#}]" ::: \
    "${str[@]}" 2>&1 | tee $OUTPUT_DIR/resnet50_${TEST_MODE}_${PRECISION}.log
    fi
elif [[ ${FLEX_GPU_TYPE} == flex_170 ]]; then
    if [[ ${device_id} == "56c0" ]]; then
	    echo "resnet50 int8 inference on Flex series 170"
	    python eval_image_classifier_inference.py --input-graph=${PB_FILE_PATH} --batch-size=${BATCH_SIZE} --warmup-steps=10 --steps=500 --dtype ${PRECISION} ${ARGS} \
      |& tee $OUTPUT_DIR/resnet50_${TEST_MODE}_${PRECISION}.log
    fi
fi
fi
if [ ${TEST_MODE} == "accuracy" ];then
  value=$(cat resnet50_${TEST_MODE}_${PRECISION}.log | grep "(Top1 accuracy, Top5 accuracy) " | tail -n 1 | sed -e "s/.*(//" | sed -e "s/,.*//")
  key="accuracy"
  unit=""
elif [ ${TEST_MODE} == "inference" ];then 
    if [[ ${device_id} == "56c0" ]]; then
      value=$(cat $OUTPUT_DIR/resnet50_${TEST_MODE}_${PRECISION}.log | grep "Throughput" | sed -e "s/.*://" | sed -e "s/ images\/sec//")
    elif [[ ${device_id} == "56c1" ]]; then
      value=$(cat $OUTPUT_DIR/resnet50_${TEST_MODE}_${PRECISION}.log | grep "Throughput" | sed -e "s/.*://" | sed -e "s/ images\/sec//" |  awk '{ sum_total += $1 } END { print sum_total }')
    key="throughput"
    unit="images/s"
  fi
fi

yaml_content=$(cat <<EOF
results:
 - key: $key
   value: $value
   unit: $unit
EOF
)

# Write the content to a YAML file

echo "$yaml_content" >  ${OUTPUT_DIR}/results.yaml
echo "YAML file created, path:$OUTPUT_DIR/results.yaml"
