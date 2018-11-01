#!/bin/bash

echo 'Running with parameters:'
echo "    FRAMEWORK: ${FRAMEWORK}"
echo "    WORKSPACE: ${WORKSPACE}"
echo "    DATASET_LOCATION: ${DATASET_LOCATION}"
echo "    CHECKPOINT_DIRECTORY: ${CHECKPOINT_DIRECTORY}"
echo "    IN_GRAPH: ${IN_GRAPH}"
echo '    Mounted volumes:'
echo "        ${BENCHMARK_SCRIPTS} mounted on: ${MOUNT_BENCHMARK}"
echo "        ${MODELS_SOURCE_DIRECTORY} mounted on: ${MOUNT_MODELS_SOURCE}"
echo "        ${DATASET_LOCATION_VOL} mounted on: ${DATASET_LOCATION}"
echo "        ${CHECKPOINT_DIRECTORY_VOL} mounted on: ${CHECKPOINT_DIRECTORY}"
echo "    SINGLE_SOCKET: ${SINGLE_SOCKET}"
echo "    MODEL_NAME: ${MODEL_NAME}"
echo "    MODE: ${MODE}"
echo "    PLATFORM: ${PLATFORM}"
echo "    BATCH_SIZE: ${BATCH_SIZE}"

## install common dependencies
apt update ; apt full-upgrade -y
apt-get install python-tk numactl -y
apt install -y libsm6 libxext6
pip install --upgrade pip
pip install requests

single_socket_arg=""
if [ ${SINGLE_SOCKET} == "true" ]; then
    single_socket_arg="--single-socket"
fi

RUN_SCRIPT_PATH="common/${FRAMEWORK}/run_tf_benchmark.py"

DIR=${WORKSPACE}/${MODEL_NAME}/${PLATFORM}/${MODE}
LOG_OUTPUT=${WORKSPACE}/logs
if [ ! -d "${LOG_OUTPUT}" ];then
    mkdir ${LOG_OUTPUT}
fi

# Common execution command used by all models
function run_model() {
    # Navigate to the main benchmark directory before executing the script,
    # since the scripts use the benchmark/common scripts as well.
    cd ${MOUNT_BENCHMARK}

    # Start benchm marking
    eval ${CMD} 2>&1 | tee ${THROUGHPUT_LOGFILE}

    echo "PYTHONPATH: ${PYTHONPATH}" | tee -a ${THROUGHPUT_LOGFILE}
    echo "RUNCMD: ${CMD} " | tee -a ${THROUGHPUT_LOGFILE}
    echo "Batch Size: ${BATCH_SIZE}" | tee -a ${THROUGHPUT_LOGFILE}
    echo "Ran inference with batch size ${BATCH_SIZE} for throughput" | tee -a ${THROUGHPUT_LOGFILE}
    echo "Log location: ${THROUGHPUT_LOGFILE}"

}

# NCF model
function ncf() {
    # For nfc, if dataset location is empty, script downloads dataset at given location.
    if [ ! -d "${DATASET_LOCATION}" ];then
        mkdir -p /dataset
    fi

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_MODELS_SOURCE}
    pip install -r ${MOUNT_MODELS_SOURCE}/official/requirements.txt

    CMD="python ${RUN_SCRIPT_PATH} \
    --framework=${FRAMEWORK} \
    --model-name=${MODEL_NAME} \
    --platform=${PLATFORM} \
    --mode=${MODE} \
    --model-source-dir=${MOUNT_MODELS_SOURCE} \
    --batch-size=${BATCH_SIZE} \
    ${single_socket_arg} \
    --data-location=${DATASET_LOCATION} \
    --checkpoint=${CHECKPOINT_DIRECTORY} \
    --verbose"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
}

# SSD-MobileNet models
function ssd_mobilenet() {
    if [ ${MODE} == "inference" ] && [ ${PLATFORM} == "fp32" ]; then
        # install dependencies
        # TODO: add requirements.txt and do pip  - Dina to add.
        pip install --user Cython
        pip install --user contextlib2
        pip install --user pillow
        pip install --user lxml
        pip install --user jupyter
        pip install --user matplotlib

        original_dir=$(pwd)
        cd "${MOUNT_MODELS_SOURCE}/research"

        # install protoc, if necessary, then compile protoc files
        if [ ! -f "bin/protoc" ]; then
            echo "protoc not found, installing protoc 3.0.0"
            apt-get -y install wget
            wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
            unzip -f -o protobuf.zip
            rm protobuf.zip
        else
            echo "protoc already found"
        fi

        echo "Compiling protoc files"
        ./bin/protoc object_detection/protos/*.proto --python_out=.

        export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

        cd $original_dir
        CMD="python ${RUN_SCRIPT_PATH} \
        --framework=${FRAMEWORK} \
        --model-name=${MODEL_NAME} \
        --platform=${PLATFORM} \
        --mode=${MODE} \
        --model-source-dir=${MOUNT_MODELS_SOURCE} \
        --batch-size=${BATCH_SIZE} \
        ${single_socket_arg} \
        --data-location=${DATASET_LOCATION} \
        --in-graph=${IN_GRAPH} \
        --verbose"

        CMD=${CMD} run_model
    else
        echo "MODE:${MODE} and PLATFORM=${PLATFORM} not supported"
    fi
}

THROUGHPUT_LOGFILE=${LOG_OUTPUT}/benchmark_${MODEL_NAME}_${MODE}_throughput.log
echo 'Log output location: ${THROUGHPUT_LOGFILE}'

MODEL_NAME=`echo ${MODEL_NAME} | tr 'A-Z' 'a-z'`
if [ ${MODEL_NAME} == "ncf" ]; then
    ncf
elif [ ${MODEL_NAME} == "ssd-mobilenet" ]; then
    ssd_mobilenet
else
    echo "Unsupported model: ${MODEL_NAME}"
    exit 1
fi


