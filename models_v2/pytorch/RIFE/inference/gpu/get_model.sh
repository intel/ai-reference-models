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
#  - download required version of the RIFE model into $(pwd) and patch it
#  - download specified weights for RIFE into $(pwd)/train_log

set -e
echo Running get_model.sh with args $@.
TIMEOUT=${TIMEOUT:-20}
RIFE_PUBLIC_REPO=${RIFE_PUBLIC_REPO:-https://github.com/megvii-research/ECCV2022-RIFE}
RIFE_FULL_SHA=${RIFE_FULL_SHA:-9a195b79f83a2e634f91369dfc9cb59cdd91fa84}

# These weights are identical to the pre-trained model documented in the
# RIFE reference implementation - the ArXiV version for IFNet
# - https://github.com/megvii-research/ECCV2022-RIFE/tree/9a195b79f83a2e634f91369dfc9cb59cdd91fa84#evaluation
# - https://drive.google.com/file/d/1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX/view?usp=sharing
WEIGHTS_GOOGLEDRIVE_ARCHIVE_ID=${WEIGHTS_GOOGLEDRIVE_ARCHIVE_ID:-"1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX"}

RIFE_ARCHIVE_PATH=${RIFE_PUBLIC_REPO}/archive/${RIFE_FULL_SHA}.zip

# Fetch model source code, and apply performance fix
MODELS_TARGET_DIR=${1:-model}
if [[ -d ${MODELS_TARGET_DIR} ]]
then
    echo "The directory ${MODELS_TARGET_DIR} already exists. Skipping model download and performance patches for Intel XPUs"
    echo "To force the script to reinstall, remove the ${MODELS_TARGET_DIR} directory manually and rerun."
else
    WORKING_DIR=$(pwd)
    wget -T ${TIMEOUT} -O models.zip ${RIFE_ARCHIVE_PATH}
    unzip -o models.zip
    mv ECCV2022-RIFE-${RIFE_FULL_SHA} ${MODELS_TARGET_DIR}
    cd ${MODELS_TARGET_DIR}
    patch -p6 < ${WORKING_DIR}/patches/0001-XPU-support-for-RIFE-IFNET-models.patch
    cd ${WORKING_DIR}
    rm models.zip
fi

# Download weights
CUSTOM_WEIGHTS_PATH=$2
WEIGHTS_TARGET_DIR=train_log

if [[ ! -z ${CUSTOM_WEIGHTS_PATH} ]] && [[ ! -f ${CUSTOM_WEIGHTS_PATH} ]]
then
    echo "Warning: get_model.sh: Custom weights path ${CUSTOM_WEIGHTS_PATH} not found."
    echo "    This may result in errors in running the application."
elif [[ -d ${WEIGHTS_TARGET_DIR} ]]
then
    echo "The directory ${WEIGHTS_TARGET_DIR} already exists. Skipping default weights download."
    echo "To force the script to reinstall, remove the \"${WEIGHTS_TARGET_DIR}\" directory manually and rerun."
else
    #Use gdown package to download archive from Google Drive share
    #More info regarding the tool is here: https://github.com/wkentaro/gdown
    echo "Downloading default weights."
    gdown ${WEIGHTS_GOOGLEDRIVE_ARCHIVE_ID}
    mkdir -p ${WEIGHTS_TARGET_DIR}
    unzip -o RIFE_trained_v6.zip
    rm -rf ./__MACOSX
    rm RIFE_trained_v6.zip
fi
