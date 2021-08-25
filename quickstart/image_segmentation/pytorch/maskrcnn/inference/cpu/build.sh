#!/usr/bin/env bash
#
# Copyright (c) 2021 Intel Corporation
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

PACKAGE_NAME=pytorch-spr-maskrcnn-inference
DOCKERFILE=pytorch-spr-maskrcnn-inference.Dockerfile
PYTORCH_BASE_IMAGE=${PYTORCH_BASE_IMAGE:-model-zoo}
PYTORCH_BASE_TAG=${PYTORCH_BASE_TAG:-pytorch-ipex-spr}
IMAGE_NAME=${IMAGE_NAME:-model-zoo:pytorch-spr-maskrcnn-inference}

if [ "$(docker images -q ${PYTORCH_BASE_IMAGE}:${PYTORCH_BASE_TAG})" == "" ]; then
  echo "The Intel(R) Extension for PyTorch container (${PYTORCH_BASE_IMAGE}:${PYTORCH_BASE_TAG}) was not found."
  echo "This container is required, as it is used as the base for building the maskrcnn inference container."
  echo "Please download the IPEX container package and build the image and then retry this build."
  exit 1
fi

docker build --build-arg PYTORCH_IMAGE=${PYTORCH_BASE_IMAGE} \
             --build-arg PYTORCH_TAG=${PYTORCH_BASE_TAG} \
             --build-arg PACKAGE_NAME=$PACKAGE_NAME \
             --build-arg MODEL_WORKSPACE=/workspace \
             --build-arg MASKRCNN_DIR=/workspace/pytorch-spr-maskrcnn-inference/models/maskrcnn \
             --build-arg http_proxy=$http_proxy \
             --build-arg https_proxy=$https_proxy  \
             --build-arg no_proxy=$no_proxy \
             -t $IMAGE_NAME \
             -f $DOCKERFILE .
