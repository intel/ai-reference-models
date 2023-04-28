#!/usr/bin/env bash
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

set -e
GPU_TYPE=$1

PYTORCH_BASE_IMAGE=${PYTORCH_BASE_IMAGE:-intel/intel-extension-for-pytorch}

if [[ $GPU_TYPE == max-series ]];then

    IMAGE_NAME=${IMAGE_NAME:-intel/image-recognition:pytorch-max-gpu-resnet50v1-5-inference}
    PYTORCH_BASE_TAG=${PYTORCH_BASE_TAG:-xpu-max}
    docker build \
    --build-arg PACKAGE_DIR=model_packages \
    --build-arg PACKAGE_NAME=pytorch-max-series-resnet50v1-5-inference \
    --build-arg MODEL_WORKSPACE=/workspace \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    --build-arg no_proxy=$no_proxy \
    --build-arg PYTORCH_BASE_IMAGE=${PYTORCH_BASE_IMAGE} \
    --build-arg PYTORCH_BASE_TAG=${PYTORCH_BASE_TAG} \
    -t $IMAGE_NAME \
    -f pytorch-max-series-resnet50v1-5-inference.Dockerfile .
elif [[ $GPU_TYPE == flex-series ]];then
    IMAGE_NAME=${IMAGE_NAME:-intel/image-recognition:pytorch-flex-gpu-resnet50v1-5-multi-card-inference}
    PYTORCH_BASE_TAG=${PYTORCH_BASE_TAG:-xpu-flex}
    docker build \
    --build-arg PACKAGE_DIR=model_packages \
    --build-arg PACKAGE_NAME=pytorch-flex-series-resnet50v1-5-multi-card-inference \
    --build-arg MODEL_WORKSPACE=/workspace \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    --build-arg no_proxy=$no_proxy \
    --build-arg PYTORCH_BASE_IMAGE=${PYTORCH_BASE_IMAGE} \
    --build-arg PYTORCH_BASE_TAG=${PYTORCH_BASE_TAG} \
    -t $IMAGE_NAME \
    -f pytorch-flex-series-resnet50v1-5-multi-card-inference.Dockerfile .
else
    echo "Only flex-series or max-series GPU platforms supported"
    exit 1
fi
