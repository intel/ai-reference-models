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
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[NUM_DEVICES]=${NUM_DEVICES}

MULTI_NODE=${MULTI_NODE:-False}
for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}

  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

if [[ "${PLATFORM}" == "Max" ]]; then
    BATCH_SIZE=${BATCH_SIZE:-256}
    PRECISION=${PRECISION:-BF16}
    NUM_ITERATIONS=${NUM_ITERATIONS:-20}
elif [[ "${PLATFORM}" == "Arc" ]]; then
    if [[ "${MULTI_TILE}" == "True" || "${MULTI_NODE}" == "True" ]]; then
        echo "Arc not support multinode/multitile"
        exit 1
    fi
    BATCH_SIZE=${BATCH_SIZE:-256}
    PRECISION=${PRECISION:-BF16}
    NUM_ITERATIONS=${NUM_ITERATIONS:-20}

fi

if [[ "${MULTI_NODE}" == "True" ]]; then
    MULTI_TILE=True
    declare -A input_envs
    multi_node_envs[HOSTFILE]=${HOSTFILE}
    multi_node_envs[MASTER_ADDR]=${MASTER_ADDR}
    multi_node_envs[SSH_PORT]=${SSH_PORT}
    multi_node_envs[NUM_PROCESS]=${NUM_PROCESS:-4}
    multi_node_envs[NUM_PROCESS_PER_NODE]=${NUM_PROCESS_PER_NODE:-2}

    for i in "${multi_node_envs[@]}"; do
        var_name=$i
        env_param=${multi_node_envs[$i]}

    if [[ -z $env_param ]]; then
        echo "The required environment variable $var_name is not set" >&2
        exit 1
    fi
    done
    if [[ ! -f "${HOSTFILE}" ]]; then
        echo "The HOSTFILE '${HOSTFILE}' does not exist"
        exit 1
    fi

    if [[ "${NUM_PROCCESS_PER_NODE}" -gt ${NUM_PROCESS} ]];then
        echo "NUM_PROCESS_PER_NODE cannot be greater than NUM_PROCESS"
        exit 1
    fi
fi

if [[ "${PRECISION}" == "BF16" ]]; then
    flag="--bf16 1 "
elif [[ "${PRECISION}" == "FP32" ]]; then
    flag=""
elif [[ "${PRECISION}" == "TF32" ]]; then
    flag="--tf32 1 "
else
    echo -e "Invalid input! Only BF16 FP32 TF32 are supported."
    exit 1
fi

if [[ "${MULTI_NODE}" == "True" ]]; then
    master_ip_flag="--dist-url ${MASTER_ADDR}"
    port_flag="--dist-port ${SSH_PORT}"
    num_process=${NUM_PROCESS}
    ppn=${NUM_PROCESS_PER_NODE}
    hostfile="-f ${HOSTFILE}"
    export FI_TCP_IFACE=${FI_TCP_IFACE:-eno0}
    export I_MPI_HYDRA_IFACE=${FI_TCP_IFACE}
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    export OMPI_MCA_tl_tcp_if_exclude="lo,docker0"
    export CCL_ATL_TRANSPORT=ofi
    export FI_PROVIDER=TCP
else 
    master_ip_flag=""
    port_flag=""
    num_process=${NUM_DEVICES}
    ppn=${NUM_DEVICES}
    hostfile=""
fi

echo "resnet50 ${PRECISION} training MultiTile=${MULTI_TILE} NumDevices=${NUM_DEVICES} BS=${BATCH_SIZE} Iter=${NUM_ITERATIONS}"


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
echo " NUM_DEVICES: ${NUM_DEVICES}"

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

modelname=resnet50

if [[ ${NUM_DEVICES} == 1 ]]; then
    rm ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0_raw.log
    python main.py \
        -a resnet50 \
        -b ${BATCH_SIZE} \
        --xpu 0 \
        ${DATASET_DIR} \
        --num-iterations ${NUM_ITERATIONS} \
        $flag 2>&1 | tee ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0_raw.log
    python common/parse_result.py -m $modelname -l ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0_raw.log -b ${BATCH_SIZE}
    throughput=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0.log | grep Performance | awk -F ' ' '{print $2}')
    throughput_unit=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0.log | grep Performance | awk -F ' ' '{print $3}')
    latency=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0.log | grep Latency | awk -F ' ' '{print $2}')
    acc=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0.log | grep Accuracy | awk -F ' ' '{print $3}')
    acc_unit=$(cat ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_t0.log | grep Accuracy | awk -F ' ' '{print $2}')
else
    rm ${OUTPUT_DIR}/${modelname}_${PRECISION}_train_raw.log
    if [[ ${CONTAINER} == "Singularity" ]]; then
        mpiexec -np ${NUM_PROCESS} -ppn ${NUM_PROCESS_PER_NODE} --hostfile ${HOSTFILE} --prepend-rank --map-by node python -u /workspace/pytorch-max-series-resnet50v1-5-training/models/main.py \
	    -a resnet50 \
	    -b ${BATCH_SIZE} \
	    --xpu 0 \
	    --dummy \
	    --num-iterations ${NUM_ITERATIONS} \
	    --bucket-cap 200 --disable-broadcast-buffers ${flag} --large-first-bucket --use-gradient-as-bucket-view \
	    --seed 123 \
	    $master_ip_flag \
	    $port_flag
    else
        mpiexec -np ${num_process} -ppn ${ppn} --prepend-rank ${hostfile} python -u main.py \
	    -a resnet50 \
	    -b ${BATCH_SIZE} \
	    --xpu 0 \
	    --dummy \
	    --num-iterations ${NUM_ITERATIONS} \
	    --bucket-cap 200 --disable-broadcast-buffers ${flag} --large-first-bucket --use-gradient-as-bucket-view \
	    --seed 123 \ 
	    $master_ip_flag \
	    $port_flag 2>&1 | tee ${OUTPUT_DIR}/ddp-${modelname}_${PRECISION}_train_raw.log 
    fi
    python common/parse_result.py -m $modelname --ddp -l ${OUTPUT_DIR}/ddp-${modelname}_${PRECISION}_train_raw.log -b ${BATCH_SIZE}
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
echo "$yaml_content" >  ${OUTPUT_DIR}/results.yaml
echo "YAML file created."
