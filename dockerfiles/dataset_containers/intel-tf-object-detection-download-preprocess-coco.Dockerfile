# Copyright (c) 2020 Intel Corporation
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
ARG TENSORFLOW_TAG

FROM ${TENSORFLOW_IMAGE}:${TENSORFLOW_TAG}


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing python-tk libsm6 libxext6 -y && \
    pip install requests


ARG TF_MODELS_BRANCH
ARG CODE_DIR=/tensorflow/models

ENV TF_MODELS_DIR=${CODE_DIR}

RUN apt-get install -y git && \
    git clone https://github.com/tensorflow/models.git ${CODE_DIR} && \
    ( cd ${CODE_DIR} && git checkout ${TF_MODELS_BRANCH} )


# Note pycocotools has to be install after the other requirements
RUN pip install numpy==1.17.4 Cython contextlib2 pillow>=6.2. lxml jupyter matplotlib && \
    pip install pycocotools


ARG TF_MODELS_DIR=/tensorflow/models

# Downloads protoc and runs it for object detection
RUN cd ${TF_MODELS_DIR}/research && \
    apt-get install -y wget unzip && \
    wget --quiet -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip && \
    unzip -o protobuf.zip && \
    rm protobuf.zip && \
    ./bin/protoc object_detection/protos/*.proto --python_out=. && \
    apt-get remove -y wget unzip


ARG PACKAGE_DIR=model_packages
ARG PACKAGE_NAME

# /workspace and below needs to be owned by root:root rather than the current UID:GID
# this allows the default user (root) to work in k8s single-node, multi-node
ADD --chown=0:0 ${PACKAGE_DIR}/${PACKAGE_NAME}.tar.gz /workspace
RUN chown -R root /workspace/${PACKAGE_NAME} && chgrp -R root /workspace/${PACKAGE_NAME} && chmod -R g+s+w /workspace/${PACKAGE_NAME}

WORKDIR /workspace/${PACKAGE_NAME}


RUN apt-get install -y wget unzip

CMD examples/download_and_preprocess_coco.sh

