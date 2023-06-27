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

RUN apt-get update && \
    apt-get install -y parallel
RUN apt-get install -y pciutils

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing numactl

ARG PY_VERSION=3.10

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    python${PY_VERSION}-dev

RUN pip install opencv-python

# Note pycocotools has to be install after the other requirements
RUN pip install \
        Cython \
        contextlib2 \
        jupyter \
        lxml \
        matplotlib \
        numpy>=1.17.4 \
        'pillow>=9.3.0'  \
        pycocotools \
        opencv-python-headless \
        pandas 

WORKDIR /workspace/pytorch-flex-series-ssd-mobilenet-inference

ARG VOC_LABELS_URL

RUN mkdir labels && \ 
    wget ${VOC_LABELS_URL} -O labels/voc-model-labels.txt 

COPY quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/README.md README.md
COPY models/object_detection/pytorch/ssd-mobilenet/inference/gpu models/object_detection/pytorch/ssd-mobilenet/inference/gpu 
COPY quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/inference_with_dummy_data.sh quickstart/inference_with_dummy_data.sh
COPY quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/flex_multi_card_batch_inference.sh quickstart/flex_multi_card_batch_inference.sh
COPY quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/flex_multi_card_online_inference.sh quickstart/flex_multi_card_online_inference.sh

COPY LICENSE license/LICENSE
COPY third_party license/third_party
