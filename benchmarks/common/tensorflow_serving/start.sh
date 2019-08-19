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
#!/usr/bin/env bash
set -e
set -x

echo 'Running with parameters:'
echo "    USE_CASE: ${USE_CASE}"
echo "    FRAMEWORK: ${FRAMEWORK}"
echo "    WORKSPACE: ${WORKSPACE}"
echo "    IN_GRAPH: ${IN_GRAPH}"
echo "    MODEL_NAME: ${MODEL_NAME}"
echo "    MODE: ${MODE}"
echo "    PRECISION: ${PRECISION}"
echo "    BATCH_SIZE: ${BATCH_SIZE}"
echo "    BENCHMARK_ONLY: ${BENCHMARK_ONLY}"
echo "    ACCURACY_ONLY: ${ACCURACY_ONLY}"
echo "    OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "    NUM_INTRA_THREADS: ${NUM_INTRA_THREADS}"
echo "    NUM_INTER_THREADS: ${NUM_INTER_THREADS}"
echo "    OUTPUT_DIR: ${OUTPUT_DIR}"
echo "    TF_SERVING_VERSION: ${TF_SERVING_VERSION}"
echo "    DATASET_LOCATION: ${DATASET_LOCATION}"


if [ ${ACCURACY_ONLY} == "True" ]; then
    echo "Accuracy is not supported with Tensorflow Serving"
    exit 1
fi

WORKDIR=workspace

if [ -d ${WORKDIR} ]; then
    rm -rf ${WORKDIR}
fi

pushd $(pwd)

mkdir -p ${WORKDIR}
cd ${WORKDIR}

# Check docker
if ! [[ $(which docker) && $(docker --version) ]]; then
    echo "Docker not found, please install docker to proceed."
    exit 1
fi

# Check for pip
if ! [[ $(which pip) && $(pip --version) ]]; then
    echo "pip not found, please install pip to proceed."
    exit 1
fi

timestamp=`date +%Y%m%d_%H%M%S`
LOG_FILENAME="benchmark_${MODEL_NAME}_${MODE}_${PRECISION}_${timestamp}.log"
if [ ! -d "${OUTPUT_DIR}" ]; then
  mkdir ${OUTPUT_DIR}
fi

function install_protoc() {
  pushd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"

  # install protoc, if necessary, then compile protoc files
  if [ ! -f "bin/protoc" ]; then
    install_location=$1
    echo "protoc not found, installing protoc from ${install_location}"
    apt-get -y install wget unzip
    wget -O protobuf.zip ${install_location}
    unzip -o protobuf.zip
    rm protobuf.zip
  else
    echo "protoc already found"
  fi

  echo "Compiling protoc files"
  ./bin/protoc object_detection/protos/*.proto --python_out=.
  popd
}

MKL_IMAGE_TAG=tensorflow/serving:latest-mkl

# Build Tensorflow Serving docker image
echo "Building tensorflow serving image..."
echo "First time it takes few minutes to build images, consecutive builds are much faster"

TF_SERVING_VERSION=${TF_SERVING_VERSION} MKL_IMAGE_TAG=${MKL_IMAGE_TAG} bash ${WORKSPACE}/build_tfserving_image.sh

function docker_run(){
    docker run \
        --name=${CONTAINER_NAME} \
        --rm \
        -d \
        -p 8500:8500 \
        -p 8501:8501 \
        -v /tmp:/models/${model_name} \
        -e MODEL_NAME=${model_name} \
        -e OMP_NUM_THREADS=${OMP_NUM_THREADS} \
        -e TENSORFLOW_INTER_OP_PARALLELISM=${NUM_INTER_THREADS} \
        -e TENSORFLOW_INTRA_OP_PARALLELISM=${NUM_INTRA_THREADS} \
        ${MKL_IMAGE_TAG}
}


function resnet50_or_inceptionv3(){
    # Setup virtual env
    pip install virtualenv
    virtualenv venv
    source venv/bin/activate
    pip install requests tensorflow-serving-api

    # cd to image recognition tfserving scripts
    cd ${WORKSPACE}/../../${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/${MODE}/${PRECISION}

    # by default converted model is saved at /tmp/1
    rm -rf /tmp/1

    # convert pretrained model to savedmodel
    python model_graph_to_saved_model.py --import_path ${IN_GRAPH}

    RUNNING=$(docker ps --filter="expose=8501/tcp" -q | xargs)
    if [[ -n ${RUNNING} ]]; then
        docker rm -f ${RUNNING}
    fi

    CONTAINER_NAME=tfserving_${RANDOM}
    model_name=${MODEL_NAME}
    # Run container
    MKL_IMAGE_TAG=${MKL_IMAGE_TAG} CONTAINER_NAME=${CONTAINER_NAME} docker_run

    # Test
    python image_recognition_client.py --model ${MODEL_NAME}


    if [ ${BATCH_SIZE} == 1 ];then
        # Test Average latency
        python image_recognition_benchmark.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME}
    else
        # Test max throughput
        python image_recognition_benchmark.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME}
    fi

    # Clean up
    docker rm -f ${CONTAINER_NAME}
}

function rfcn_or_ssd_mobilenet(){
    # Setup virtual env
    pip install virtualenv
    virtualenv venv
    source venv/bin/activate

    cd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"
    # install protoc v3.3.0, if necessary, then compile protoc files
    install_protoc "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip"
    export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/slim

    # cd to object detection tfserving scripts
    cd ${WORKSPACE}/../../${USE_CASE}/${FRAMEWORK}

    # install dependencies
    pip install -r requirements.txt

    saved_model_dir=/tmp/1

    if [ -d ${saved_model_dir} ]; then
        rm -rf ${saved_model_dir}
    fi
    mkdir -p ${saved_model_dir}

    cp ${IN_GRAPH} ${saved_model_dir}

    RUNNING=$(docker ps --filter="expose=8500/tcp" -q | xargs)
    if [[ -n ${RUNNING} ]]; then
        docker rm -f ${RUNNING}
    fi

    CONTAINER_NAME=tfserving_${RANDOM}

    # use the defined protocol value in the env var PROTOCOL_NAME or GRPC as the default value
    PROTOCOL_NAME=${PROTOCOL_NAME:-"grpc"}

    if [ ${MODEL_NAME} == "ssd-mobilenet" ]; then
        model_name="ssdmobilenet"
    else
        model_name=${MODEL_NAME}
    fi

    # Run container
    MKL_IMAGE_TAG=${MKL_IMAGE_TAG} CONTAINER_NAME=${CONTAINER_NAME} docker_run

    # Test
    python object_detection_benchmark.py -i ${DATASET_LOCATION} -m ${model_name} -p ${PROTOCOL_NAME}

    # Clean up
    docker rm -f ${CONTAINER_NAME}
}


LOGFILE=${OUTPUT_DIR}/${LOG_FILENAME}

MODEL_NAME=$(echo ${MODEL_NAME} | tr 'A-Z' 'a-z')
if [ ${MODEL_NAME} == "inceptionv3" ] || [ ${MODEL_NAME} == "resnet50" ] && [ ${PRECISION} == "fp32" ]; then
  resnet50_or_inceptionv3 | tee -a ${LOGFILE}
elif [ ${MODEL_NAME} == "rfcn" ] || [ ${MODEL_NAME} == "ssd-mobilenet" ] && [ ${PRECISION} == "fp32" ]; then
  rfcn_or_ssd_mobilenet | tee -a ${LOGFILE}
else
  echo "Unsupported Model: ${MODEL_NAME} or Precision: ${PRECISION}"
  exit 1
fi

popd

# Clean up work directory
rm -rf ${WORKDIR}

echo "Log output location: ${LOGFILE}" | tee -a ${LOGFILE}
