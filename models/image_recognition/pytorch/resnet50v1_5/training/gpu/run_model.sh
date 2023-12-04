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
input_envs[MULTI_TILE]=${MULTI_TILE}
input_envs[PLATFORM]=${PLATFORM}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}

  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

OUTPUT_DIR=${OUTPUT_DIR:-$PWD}

if [[ "${PLATFORM}" == "PVC" ]]; then
    BATCH_SIZE=${BATCH_SIZE:-256}
    PRECISION=${PRECISION:-bf16}
    NUM_ITERATIONS=${NUM_ITERATIONS:-20}
elif [[ "${PLATFORM}" == "ATS-M" ]]; then
    if [[ "${MULTI_TILE}" == "True" ]]; then
        echo "ATS-M not support multitile"
        exit 1
    fi
    BATCH_SIZE=${BATCH_SIZE:-128}
    PRECISION=${PRECISION:-bf16}
    NUM_ITERATIONS=${NUM_ITERATIONS:-20}
fi

if [[ "${PRECISION}" == "bf16" ]]; then
    flag="--bf16 1 "
elif [[ "${PRECISION}" == "fp32" ]]; then
    flag=""
elif [[ "${PRECISION}" == "tf32" ]]; then
    flag="--tf32 1 "
else
    echo -e "Invalid input! Only bf16 fp32 tf32 are supported."
    exit 1
fi

echo "resnet50 ${PRECISION} training MultiTile=${MULTI_TILE} BS=${BATCH_SIZE} Iter=${NUM_ITERATIONS}"


if [[ ! -d "${DATASET_DIR}" ]] && [[ "${MULTI_TILE}" != "True" ]]; then
    echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
    exit 1
fi

echo 'Running with parameters:'
echo " PLATFORM: ${PLATFORM}"
echo " DATASET_PATH: ${DATASET_DIR}"
echo " OUTPUT_DIR: ${OUTPUT_DIR}"
echo " PRECISION: ${PRECISION}"
echo " BATCH_SIZE: ${BATCH_SIZE}"
echo " NUM_ITERATIONS: ${NUM_ITERATIONS}"
echo " MULTI_TILE: ${MULTI_TILE}"



# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR


modelname=resnet50

if [[ ${MULTI_TILE} == "False" ]]; then
    rm ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0_raw.log
    python main.py \
        -a resnet50 \
        -b ${BATCH_SIZE} \
        --xpu 0 \
        ${DATASET_DIR} \
        --num-iterations ${NUM_ITERATIONS} \
        $flag 2>&1 | tee ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0_raw.log
    python ../../../../../common/pytorch/parse_result.py -t por -l ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0_raw.log -b ${BATCH_SIZE}
    throughput=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0.log | grep Performance | awk -F ' ' '{print $2}')
    throughput_unit=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0.log | grep Performance | awk -F ' ' '{print $3}')
    latency=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0.log | grep Latency | awk -F ' ' '{print $2}')
    acc=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0.log | grep Accuracy | awk -F ' ' '{print $3}')
    acc_unit=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0.log | grep Accuracy | awk -F ' ' '{print $2}')
else
    rm ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_raw.log
    mpiexec -np 2 -ppn 2 --prepend-rank python -u main.py -a resnet50 -b ${BATCH_SIZE} --xpu 0 --dummy --num-iterations ${NUM_ITERATIONS} --bucket-cap 200 --disable-broadcast-buffers ${flag} --large-first-bucket --use-gradient-as-bucket-view --seed 123 2>&1 | tee ${OUTPUT_DIR}/ddp-${modelname}_${PRECISION}_train_raw.log 
    python ../../../../../common/pytorch/parse_result.py -t ddp -l ${OUTPUT_DIR}/ddp-${modelname}_${PRECISION}_train_raw.log -b ${BATCH_SIZE}
    throughput=$(cat ${OUTPUT_DIR}/ddp-${modelname}_${PRECISION}_train.log | grep "Sum Performance" | awk -F ' ' '{print $3}')
    throughput_unit=$(cat ${OUTPUT_DIR}/ddp-${modelname}_${PRECISION}_train.log | grep "Sum Performance" | awk -F ' ' '{print $4}')
    latency=$(cat ${OUTPUT_DIR}/ddp-${modelname}_${PRECISION}_train.log | grep Latency | awk -F ' ' '{print $2}')
    acc=$(cat ${OUTPUT_DIR}/ddp-${modelname}_${PRECISION}_train.log | grep Accuracy | awk -F ' ' '{print $3}')
    acc_unit=$(cat ${OUTPUT_DIR}/ddp-${modelname}_${PRECISION}_train.log | grep Accuracy | awk -F ' ' '{print $2}')
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
echo "$yaml_content" >  ./results.yaml
echo "YAML file created."

