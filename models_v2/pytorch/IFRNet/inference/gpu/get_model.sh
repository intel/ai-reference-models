# Copyright (c) 2023-2024 Intel Corporation
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

#!/usr/bin/env bash

# This script does the following
#  - download required version of the IFRNet model into $(pwd) and patch it
#  - download specified weights for IFRNet into $(pwd)/checkpoints

set -e
echo Running get_model.sh with args $@.
TIMEOUT=${TIMEOUT:-20}
IFRNET_PUBLIC_REPO=${IFRNET_PUBLIC_REPO:-https://github.com/ltkong218/IFRNet}
IFRNET_FULL_SHA=${IFRNET_FULL_SHA:-b117bcafcf074b2de756b882f8a6ca02c3169bfe}

# For reliability purpose we fetch the weights from our own host.
# These weights are identical to the pre-trained model documented in the
# IFRNet reference implementation (linked below).
# - https://github.com/ltkong218/IFRNet/tree/b117bcafcf074b2de756b882f8a6ca02c3169bfe?tab=readme-ov-file#download-pre-trained-models-and-play-with-demos
# MD5Sum: e15dfd3225e98d1e340c8b3a726638fd
WEIGHTS_ARCHIVE_URL=${WEIGHTS_ARCHIVE_URL:-"https://storage.googleapis.com/intel-optimized-tensorflow/models/3_2/IFRNet-checkpoints.zip"}

IFRNET_ARCHIVE_PATH=${IFRNET_PUBLIC_REPO}/archive/${IFRNET_FULL_SHA}.zip

# Fetch model source code, and apply performance fix
MODELS_TARGET_DIR=${1:-models}
if [[ -d ${MODELS_TARGET_DIR} ]]
then
    echo "The directory \"models\" already exists. Skipping model download and performance patches for Intel XPUs"
    echo "To force the script to reinstall, remove the \"models\" directory manually and rerun."
else
    WORKING_DIR=$(pwd)
    wget -T ${TIMEOUT} -O models.zip ${IFRNET_ARCHIVE_PATH}
    unzip -o models.zip
    mv IFRNet-${IFRNET_FULL_SHA} ${MODELS_TARGET_DIR}
    cd ${MODELS_TARGET_DIR}
    patch -p5 < ${WORKING_DIR}/patches/0001-Default-tensors-to-GPU-in-IFRNet-models-utils.py-71.patch
    cd ${WORKING_DIR}
    rm models.zip
fi

# Download weights
CUSTOM_WEIGHTS_PATH=$2
CHECKPOINTS_TARGET_DIR=checkpoints

if [[ ! -z ${CUSTOM_WEIGHTS_PATH} ]] && [[ ! -f ${CUSTOM_WEIGHTS_PATH} ]]
then
    echo "Warning: get_model.sh: Custom weights path ${CUSTOM_WEIGHTS_PATH} not found."
    echo "    This may result in errors in running the application."
elif [[ -d ${CHECKPOINTS_TARGET_DIR} ]]
then
    echo "The directory \"checkpoints\" already exists. Skipping default weights download."
    echo "To force the script to reinstall, remove the \"checkpoints\" directory manually and rerun."
else
    echo "Downloading default weights."
    wget -T ${TIMEOUT} -O checkpoints.zip ${WEIGHTS_ARCHIVE_URL}
    mkdir -p ${CHECKPOINTS_TARGET_DIR}
    unzip -o checkpoints.zip -x / -d ${CHECKPOINTS_TARGET_DIR}
    rm checkpoints.zip
fi
