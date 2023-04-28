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

TENSORFLOW_BASE_IMAGE=${TENSORFLOW_BASE_IMAGE:-intel/intel-extension-for-tensorflow}
TENSORFLOW_BASE_TAG=${TENSORFLOW_BASE_TAG:-gpu-flex}
IMAGE_NAME=${IMAGE_NAME:-intel/object-detection:tf-flex-gpu-ssd-mobilenet-multi-card-inference}

docker build \
    --build-arg PACKAGE_DIR=model_packages \
    --build-arg PACKAGE_NAME=tf-flex-series-ssd-mobilenet-multi-card-inference \
    --build-arg MODEL_WORKSPACE=/workspace \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    --build-arg no_proxy=$no_proxy \
    --build-arg TENSORFLOW_BASE_IMAGE=${TENSORFLOW_BASE_IMAGE} \
    --build-arg TENSORFLOW_BASE_TAG=${TENSORFLOW_BASE_TAG} \
    -t $IMAGE_NAME \
    -f tf-flex-series-ssd-mobilenet-multi-card-inference.Dockerfile .

