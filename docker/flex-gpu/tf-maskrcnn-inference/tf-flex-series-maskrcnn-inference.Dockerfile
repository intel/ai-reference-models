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

ARG BASE_IMAGE="intel/intel-extension-for-tensorflow"
ARG BASE_TAG="xpu"

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace/tf-flex-series-maskrcnn-inference

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing parallel pciutils numactl
    
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    git build-essential libssl-dev libffi-dev python3.10-dev

RUN python -m pip install opencv-python-headless pycocotools 

COPY models/image_segmentation/tensorflow/maskrcnn/inference/gpu models/image_segmentation/tensorflow/maskrcnn/inference/gpu
COPY quickstart/image_segmentation/tensorflow/maskrcnn/inference/gpu/inference.sh quickstart/inference.sh

RUN pip install git+https://github.com/NVIDIA/dllogger.git
RUN git clone https://github.com/NVIDIA/DeepLearningExamples.git && \
    cd /workspace/tf-flex-series-maskrcnn-inference/DeepLearningExamples && \
    git checkout 5be8a3cae21ee2d80e3935a4746827cb3367bcac && \
    mv /workspace/tf-flex-series-maskrcnn-inference/models/image_segmentation/tensorflow/maskrcnn/inference/gpu/EnableInference.patch . && \
    git apply EnableInference.patch && \
    cd -

COPY LICENSE license/LICENSE
COPY third_party license/third_party
