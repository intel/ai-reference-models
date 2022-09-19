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

PACKAGE_NAME=tf-spr-3d-unet-mlperf-inference
DOCKERFILE=tf-spr-3d-unet-mlperf-inference.Dockerfile
TF_SPR_BASE_IMAGE=${TF_SPR_BASE_IMAGE:-model-zoo}
TF_SPR_BASE_TAG=${TF_SPR_BASE_TAG:-tensorflow-spr}
IMAGE_NAME=${IMAGE_NAME:-model-zoo:tf-spr-3d-unet-mlperf-inference}

if [ "$(docker images -q ${TF_SPR_BASE_IMAGE}:${TF_SPR_BASE_TAG})" == "" ]; then
  echo "The Intel(R) TensorFlow SPR base container for 3D-UNet (${TF_SPR_BASE_IMAGE}:${TF_SPR_BASE_TAG}) was not found."
  echo "This container is required, as it is used as the base for building the MLPerf 3D U-Net inference container."
  echo "Please download the TensorFlow SPR container package and build the image and then retry this build."
  exit 1
fi

docker build --build-arg TENSORFLOW_IMAGE=${TF_SPR_BASE_IMAGE} \
             --build-arg TENSORFLOW_TAG=${TF_SPR_BASE_TAG} \
             --build-arg PACKAGE_NAME=$PACKAGE_NAME \
             --build-arg MODEL_WORKSPACE=/workspace \
             --build-arg http_proxy=$http_proxy \
             --build-arg https_proxy=$https_proxy  \
             --build-arg no_proxy=$no_proxy \
             -t $IMAGE_NAME \
             -f $DOCKERFILE .
