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
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

<<<<<<<< HEAD:docker/flex-gpu/tf-efficientnet-inference/tf-flex-series-efficientnet-inference.Dockerfile
ARG BASE_IMAGE="intel/intel-extension-for-tensorflow"
ARG BASE_TAG="xpu"

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace/tf-flex-series-efficientnet-inference

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing parallel pciutils numactl

RUN pip install pillow 

COPY models/image_recognition/tensorflow/efficientnet/inference/gpu/predict.py models/image_recognition/tensorflow/efficientnet/inference/gpu/predict.py 
COPY quickstart/image_recognition/tensorflow/efficientnet/inference/gpu/batch_inference.sh quickstart/batch_inference.sh

COPY LICENSE license/LICENSE
COPY third_party license/third_party
========
ARG PYT_BASE_IMAGE="intel/intel-extension-for-pytorch"
ARG PYT_BASE_TAG="2.1.10-xpu"

FROM ${PYT_BASE_IMAGE}:${PYT_BASE_TAG}

USER root

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace/pytorch-max-series-resnet50v1-5-inference/models

COPY models_v2/pytorch/resnet50v1_5/inference/gpu .
COPY models_v2/common common 

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party

USER $USER
>>>>>>>> r3.1:docker/max-gpu/pytorch-resnet50v1-5-inference/pytorch-max-series-resnet50v1-5-inference.Dockerfile
