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
input_envs[MULTI_TILE]=${MULTI_TILE}
input_envs[PLATFORM]=${PLATFORM}
input_envs[BERT_WEIGHT]=${BERT_WEIGHT}
input_envs[DATASET_DIR]=${DATASET_DIR}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[NUM_DEVICES]=${NUM_DEVICES}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}

  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

if [[ ! -d "${BERT_WEIGHT}" ]]; then
  echo "The BERT_WEIGHT '${BERT_WEIGHT}' does not exist"
  exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

mkdir -p ${OUTPUT_DIR}

if [[ "${PLATFORM}" == "Max" ]]; then
    BATCH_SIZE=${BATCH_SIZE:-256}
    PRECISION=${PRECISION:-BF16}
    NUM_ITERATIONS=${NUM_ITERATIONS:--1}
elif [[ "${PLATFORM}" == "Flex" ]]; then
    echo "only support Max for platform"
elif [[ "${PLATFORM}" == "Arc" ]]; then
    if [[ "${MULTI_TILE}" == "True" ]]; then
        echo "Only support MULTI_TILE=False when in arc platform"
        exit 1
    fi
    BATCH_SIZE=${BATCH_SIZE:-64}
    PRECISION=${PRECISION:-FP16}
    NUM_ITERATIONS=${NUM_ITERATIONS:--1}
fi

# known issue
#if [[ "${MULTI_TILE}" == "True" ]]; then
#    export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
#fi

echo 'Running with parameters:'
echo " PLATFORM: ${PLATFORM}"
echo " OUTPUT_DIR: ${OUTPUT_DIR}"
echo " PRECISION: ${PRECISION}"
echo " BATCH_SIZE: ${BATCH_SIZE}"
echo " NUM_ITERATIONS: ${NUM_ITERATIONS}"
echo " MULTI_TILE: ${MULTI_TILE}"
echo " NUM_DEVICES: ${NUM_DEVICES}"

if [[ "${PRECISION}" != "BF16" ]] && [[ "${PRECISION}" != "FP32" ]] && [[ "${PRECISION}" != "FP16" ]]; then
    echo -e "Invalid input! Only BF16 FP32 FP16 are supported."
    exit 1
fi
echo "bert-large ${PRECISION} inference plain MultiTile=${MULTI_TILE} NumDevices=${NUM_DEVICES} BS=${BATCH_SIZE} Iter=${NUM_ITERATIONS}"

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

modelname=bertsquad

if [[ ${NUM_DEVICES} == 1 ]]; then
    rm ${OUTPUT_DIR}/${modelname}${PRECISION}_inf_t0_raw.log 
    bash cmd_infer.sh \
        -m bert_large \
        -b ${BATCH_SIZE} \
        -d xpu \
        -t ${PRECISION} \
        -o ${OUTPUT_DIR} \
	-s ${DATASET_DIR} \
	-w ${BERT_WEIGHT} \
        -n ${NUM_ITERATIONS} 2>&1 | tee ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_t0_raw.log
    python common/parse_result.py -m $modelname -l ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_t0_raw.log -b ${BATCH_SIZE}
    throughput=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_t0.log | grep Performance | awk -F ' ' '{print $2}')
    throughput_unit=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_t0.log | grep Performance | awk -F ' ' '{print $3}')
    latency=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_t0.log | grep Latency | awk -F ' ' '{print $2}')
    acc=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_t0.log | grep Accuracy | awk -F ' ' '{print $3}')
    acc_unit=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_t0.log | grep Accuracy | awk -F ' ' '{print $2}')
else
    rm ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_device*_raw.log
    for i in $(seq 0 $((NUM_DEVICES-1)));do
        str+=("
	    ZE_AFFINITY_MASK=${i} bash cmd_infer.sh \
	        -m bert_large \
		-b ${BATCH_SIZE} \
		-d xpu \
		-t ${PRECISION} \
		-o ${OUTPUT_DIR} \
		-n ${NUM_ITERATIONS} 2>&1 | tee ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_device${i}_raw.log
	")
    done
    parallel --lb -d, --tagstring "[{#}]" ::: "${str[@]}" 2>&1 | tee ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_${NUM_DEVICES}devices_raw.log
    
    throughput=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_${NUM_DEVICES}devices_raw.log | grep "bert_inf throughput" | awk -F ' ' '{print $4}' | awk '{ sum_total += $1 } END { printf "%.4f\n",sum_total}')
    throughput_unit="sent/s"
    latency=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_${NUM_DEVICES}devices_raw.log | grep "bert_inf latency" | awk -F ' ' '{print $4}' | awk '{ sum_total += $1 } END { print sum_total/NR }')
    acc=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_inf_${NUM_DEVICES}devices_raw.log | grep Results | awk -F 'best_f1' '{print $2}' | awk -F ' ' '{print $2}' | awk '{ sum_total += $1 } END { printf "%.3f\n",sum_total/NR }')
    acc_unit="f1"
fi

yaml_content=$(cat <<EOF
results:
 - key: throughput
   value: $throughput
   unit: $throughput_unit
 - key: latency
   value: $latency
   unit: s
 - key: accuracy
   value: $acc
   unit: $acc_unit
EOF
)

# Write the content to a YAML file
echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
