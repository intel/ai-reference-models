# Copyright (c) 2020-2021 Intel Corporation
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

ARG BASE_IMAGE="intel/intel-extension-for-pytorch"
ARG BASE_TAG="xpu-flex"

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace/pytorch-flex-series-resnet50v1-5-inference

COPY quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/README_Flex_Series.md README.md
COPY models/image_recognition/pytorch/resnet50v1_5/inference/gpu models/image_recognition/pytorch/resnet50v1_5/inference/gpu
COPY quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/inference_block_format.sh quickstart/inference_block_format.sh

COPY LICENSE license/LICENSE
COPY third_party license/third_party
