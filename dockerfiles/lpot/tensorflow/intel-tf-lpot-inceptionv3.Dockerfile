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
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG TENSORFLOW_IMAGE="intel/intel-optimized-tensorflow"

ARG TENSORFLOW_TAG="2.5.0-ubuntu-20.04"

FROM ${TENSORFLOW_IMAGE}:${TENSORFLOW_TAG}

ARG PY_VERSION="3.8"

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
        build-essential \
        python${PY_VERSION}-dev

RUN pip install lpot

ARG LPOT_SOURCE_DIR=/src/lpot
ARG LPOT_BRANCH=master

ENV LPOT_SOURCE_DIR=$LPOT_SOURCE_DIR

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y git && \
    git clone --single-branch --branch ${LPOT_BRANCH} https://github.com/intel/lpot.git ${LPOT_SOURCE_DIR}

WORKDIR ${LPOT_SOURCE_DIR}

RUN apt-get install --no-install-recommends --fix-missing -y wget

WORKDIR ${LPOT_SOURCE_DIR}/examples/tensorflow/image_recognition

RUN wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/inceptionv3_fp32_pretrained_model.pb
ENV PRETRAINED_MODEL=${PWD}/inceptionv3_fp32_pretrained_model.pb
