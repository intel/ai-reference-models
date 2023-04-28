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

TENSORFLOW_BASE_IMAGE=${TENSORFLOW_BASE_IMAGE:-intel/intel-extension-for-tensorflow}
TENSORFLOW_BASE_TAG=${TENSORFLOW_BASE_TAG:-gpu-max}
IMAGE_NAME=${IMAGE_NAME:-intel/language-modeling:tf-max-gpu-bert-large-inference}

docker build \
    --build-arg PACKAGE_DIR=model_packages \
    --build-arg PACKAGE_NAME=tf-max-series-bert-large-inference \
    --build-arg MODEL_WORKSPACE=/workspace \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    --build-arg no_proxy=$no_proxy \
    --build-arg TENSORFLOW_BASE_IMAGE=${TENSORFLOW_BASE_IMAGE} \
    --build-arg TENSORFLOW_BASE_TAG=${TENSORFLOW_BASE_TAG} \
    -t $IMAGE_NAME \
    -f tf-max-series-bert-large-inference.Dockerfile .

