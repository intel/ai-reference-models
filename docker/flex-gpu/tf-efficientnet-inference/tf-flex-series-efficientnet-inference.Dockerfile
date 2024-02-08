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

<<<<<<<< HEAD:docker/max-gpu/pytorch-resnet50v1-5-inference/pytorch-max-series-resnet50v1-5-inference.Dockerfile
ARG BASE_IMAGE="intel/intel-extension-for-pytorch"
ARG BASE_TAG="xpu-max"

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace/pytorch-max-series-resnet50v1-5-inference
COPY quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/README_Max_Series.md README.md
COPY models/image_recognition/pytorch/resnet50v1_5/inference/gpu models/image_recognition/pytorch/resnet50v1_5/inference/gpu
COPY quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/inference_block_format.sh quickstart/inference_block_format.sh

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party
========
ARG TF_BASE_IMAGE="intel/intel-extension-for-tensorflow"
ARG TF_BASE_TAG="xpu"

FROM ${TF_BASE_IMAGE}:${TF_BASE_TAG}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace/tf-flex-series-efficientnet-inference/models

RUN pip install pillow

COPY models_v2/tensorflow/efficientnet/inference/gpu . 

COPY LICENSE license/LICENSE
COPY third_party license/third_party
>>>>>>>> r3.1:docker/flex-gpu/tf-efficientnet-inference/tf-flex-series-efficientnet-inference.Dockerfile
