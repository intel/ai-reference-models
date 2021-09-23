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

ARG TENSORFLOW_IMAGE="intel/intel-optimized-tensorflow"

ARG TENSORFLOW_TAG="latest"

FROM ${TENSORFLOW_IMAGE}:${TENSORFLOW_TAG}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
        libsm6 \
        libxext6 \
        python-tk && \
    pip install requests

ARG PY_VERSION="3.8"

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
        build-essential \
        python${PY_VERSION}-dev

ARG TF_MODELS_BRANCH

ARG FETCH_PR

ARG CODE_DIR=/tensorflow/models

ENV TF_MODELS_DIR=${CODE_DIR}

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y git && \
    git clone https://github.com/tensorflow/models.git ${CODE_DIR} && \
    ( cd ${CODE_DIR} && \
    if [ ! -z "${FETCH_PR}" ]; then git fetch origin ${FETCH_PR}; fi && \
    git checkout ${TF_MODELS_BRANCH} )

# Note pycocotools has to be install after the other requirements
RUN pip install \
        Cython \
        contextlib2 \
        jupyter \
        lxml \
        matplotlib \
        numpy>=1.17.4 \
        'pillow>=8.1.2' && \
    pip install pycocotools

ARG TF_MODELS_DIR=/tensorflow/models

# Downloads protoc and runs it for object detection
RUN cd ${TF_MODELS_DIR}/research && \
    apt-get install --no-install-recommends --fix-missing -y \
        unzip \
        wget && \
    wget --quiet -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip && \
    unzip -o protobuf.zip && \
    rm protobuf.zip && \
    ./bin/protoc object_detection/protos/*.proto --python_out=. && \
    apt-get remove -y \
        unzip \
        wget && \
    apt-get autoremove -y
