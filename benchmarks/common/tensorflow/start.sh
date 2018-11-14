#!/usr/bin/env bash
#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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


echo 'Running with parameters:'
echo "    FRAMEWORK: ${FRAMEWORK}"
echo "    WORKSPACE: ${WORKSPACE}"
echo "    DATASET_LOCATION: ${DATASET_LOCATION}"
echo "    CHECKPOINT_DIRECTORY: ${CHECKPOINT_DIRECTORY}"
echo "    IN_GRAPH: ${IN_GRAPH}"
echo '    Mounted volumes:'
echo "        ${BENCHMARK_SCRIPTS} mounted on: ${MOUNT_BENCHMARK}"
echo "        ${EXTERNAL_MODELS_SOURCE_DIRECTORY} mounted on: ${MOUNT_EXTERNAL_MODELS_SOURCE}"
echo "        ${INTELAI_MODELS} mounted on: ${MOUNT_INTELAI_MODELS_SOURCE}"
echo "        ${DATASET_LOCATION_VOL} mounted on: ${DATASET_LOCATION}"
echo "        ${CHECKPOINT_DIRECTORY_VOL} mounted on: ${CHECKPOINT_DIRECTORY}"
echo "    SINGLE_SOCKET: ${SINGLE_SOCKET}"
echo "    MODEL_NAME: ${MODEL_NAME}"
echo "    MODE: ${MODE}"
echo "    PLATFORM: ${PLATFORM}"
echo "    BATCH_SIZE: ${BATCH_SIZE}"
echo "    NUM_CORES: ${NUM_CORES}"
echo "    BENCHMARK_ONLY: ${BENCHMARK_ONLY}"
echo "    ACCURACY_ONLY: ${ACCURACY_ONLY}"

## install common dependencies
apt update ; apt full-upgrade -y
apt-get install python-tk numactl -y
apt install -y libsm6 libxext6
pip install --upgrade pip
pip install requests

single_socket_arg=""
if [ ${SINGLE_SOCKET} == "True" ]; then
    single_socket_arg="--single-socket"
fi

verbose_arg=""
if [ ${VERBOSE} == "True" ]; then
    verbose_arg="--verbose"
fi

RUN_SCRIPT_PATH="common/${FRAMEWORK}/run_tf_benchmark.py"

LOG_OUTPUT=${WORKSPACE}/logs
if [ ! -d "${LOG_OUTPUT}" ];then
    mkdir ${LOG_OUTPUT}
fi

export PYTHONPATH=${PYTHONPATH}:${MOUNT_INTELAI_MODELS_SOURCE}

# Common execution command used by all models
function run_model() {
    # Navigate to the main benchmark directory before executing the script,
    # since the scripts use the benchmark/common scripts as well.
    cd ${MOUNT_BENCHMARK}

    # Start benchmarking
    eval ${CMD} 2>&1 | tee ${LOGFILE}

    echo "PYTHONPATH: ${PYTHONPATH}" | tee -a ${LOGFILE}
    echo "RUNCMD: ${CMD} " | tee -a ${LOGFILE}
    echo "Batch Size: ${BATCH_SIZE}" | tee -a ${LOGFILE}
    echo "Ran ${MODE} with batch size ${BATCH_SIZE}" | tee -a ${LOGFILE}

    LOG_LOCATION_OUTSIDE_CONTAINER="${BENCHMARK_SCRIPTS}/common/${FRAMEWORK}/logs/benchmark_${MODEL_NAME}_${MODE}.log"
    echo "Log location outside container: ${LOG_LOCATION_OUTSIDE_CONTAINER}" | tee -a ${LOGFILE}
}


# basic run command with commonly used args
CMD="python ${RUN_SCRIPT_PATH} \
--framework=${FRAMEWORK} \
--model-name=${MODEL_NAME} \
--platform=${PLATFORM} \
--mode=${MODE} \
--model-source-dir=${MOUNT_EXTERNAL_MODELS_SOURCE} \
--intelai-models=${MOUNT_INTELAI_MODELS_SOURCE} \
--num-cores=${NUM_CORES} \
--batch-size=${BATCH_SIZE} \
--data-location=${DATASET_LOCATION} \
${single_socket_arg} \
${verbose_arg}"

function install_protoc() {
    # install protoc, if necessary, then compile protoc files
    if [ ! -f "bin/protoc" ]; then
        install_location=$1
        echo "protoc not found, installing protoc from ${install_location}"
        apt-get -y install wget
        wget -O protobuf.zip ${install_location}
        unzip -o protobuf.zip
        rm protobuf.zip
    else
        echo "protoc already found"
    fi

}

# NCF model
function ncf() {
    # For nfc, if dataset location is empty, script downloads dataset at given location.
    if [ ! -d "${DATASET_LOCATION}" ];then
        mkdir -p /dataset
    fi

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    pip install -r ${MOUNT_EXTERNAL_MODELS_SOURCE}/official/requirements.txt

    CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY}"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
}

# Resnet50 model
function resnet50() {
    if [ ${MODE} == "inference" ] && [ ${PLATFORM} == "int8" ]; then
        # For accuracy, dataset location is required, see README for more information.
        if [ ! -d "${DATASET_LOCATION}" ] && [ ${ACCURACY_ONLY} == "True" ];then
            echo "No Data directory specified, accuracy will not be calculated."
            exit 1
        fi

        accuracy_only_arg=""
        if [ ${ACCURACY_ONLY} == "True" ]; then
            accuracy_only_arg="--accuracy-only"
        fi

        benchmark_only_arg=""
        if [ ${BENCHMARK_ONLY} == "True" ]; then
            benchmark_only_arg="--benchmark-only"
        fi

        export PYTHONPATH=${PYTHONPATH}:`pwd`:${MOUNT_BENCHMARK}

        CMD="${CMD} ${accuracy_only_arg} \
        ${benchmark_only_arg} \
        --in-graph=${IN_GRAPH}"

        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
        echo "MODE:${MODE} and PLATFORM=${PLATFORM} not supported"
    fi
}

# SqueezeNet model
function squeezenet() {
    if [ ${MODE} == "inference" ] && [ ${PLATFORM} == "fp32" ]; then
        CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY}"

        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
        echo "MODE:${MODE} and PLATFORM=${PLATFORM} not supported"
    fi
}

# SSD-MobileNet model
function ssd_mobilenet() {
    if [ ${MODE} == "inference" ] && [ ${PLATFORM} == "fp32" ]; then
        # install dependencies
        pip install -r "${MOUNT_BENCHMARK}/object_detection/tensorflow/ssd-mobilenet/requirements.txt"

        original_dir=$(pwd)
        cd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"

        if [ ${BATCH_SIZE} != "-1" ]; then
            echo "Warning: SSD-MobileNet inference script does not use the batch_size arg"
        fi

        # install protoc, if necessary, then compile protoc files
        install_protoc "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip"

        echo "Compiling protoc files"
        ./bin/protoc object_detection/protos/*.proto --python_out=.

        export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

        cd $original_dir

        CMD="${CMD} --in-graph=${IN_GRAPH}"
        CMD=${CMD} run_model
    else
        echo "MODE:${MODE} and PLATFORM=${PLATFORM} not supported"
    fi
}
        
# Wavenet model
function wavenet() {
    if [ ${MODE} == "inference" ] && [ ${PLATFORM} == "fp32" ]; then
        export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

        pip install -r ${MOUNT_EXTERNAL_MODELS_SOURCE}/requirements.txt

        if [[ -z "${checkpoint_name}" ]]; then
            echo "wavenet requires -- checkpoint_name arg to be defined"
            exit 1
        fi

        if [[ -z "${sample}" ]]; then
            echo "wavenet requires -- sample arg to be defined"
            exit 1
        fi

        CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} \
        --checkpoint_name=${checkpoint_name} \
        --sample=${sample}"

        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
        echo "MODE:${MODE} and PLATFORM=${PLATFORM} not supported for wavenet"
    fi
}

LOGFILE=${LOG_OUTPUT}/benchmark_${MODEL_NAME}_${MODE}_${PLATFORM}.log
echo 'Log output location: ${LOGFILE}'

MODEL_NAME=`echo ${MODEL_NAME} | tr 'A-Z' 'a-z'`
if [ ${MODEL_NAME} == "ncf" ]; then
    ncf
elif [ ${MODEL_NAME} == "resnet50" ]; then
    resnet50
elif [ ${MODEL_NAME} == "squeezenet" ]; then
    squeezenet
elif [ ${MODEL_NAME} == "ssd-mobilenet" ]; then
    ssd_mobilenet
elif [ ${MODEL_NAME} == "wavenet" ]; then
    wavenet
else
    echo "Unsupported model: ${MODEL_NAME}"
    exit 1
fi
