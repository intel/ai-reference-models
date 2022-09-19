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

ARG TENSORFLOW_IMAGE="model-zoo"

ARG TENSORFLOW_TAG="tensorflow-spr"

FROM ${TENSORFLOW_IMAGE}:${TENSORFLOW_TAG}

RUN yum update -y && \
    yum install -y \
        numactl \
        libXext \
        libSM \
        python3-tkinter && \
    pip install requests

ARG PY_VER=38
RUN yum install -y gcc gcc-c++ && \
    yum install -y python${PY_VER}-devel && \
    yum clean all

ARG TF_MODELS_BRANCH="f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc"

ARG FETCH_PR

ARG CODE_DIR="/workspace/tf_models"

ENV TF_MODELS_DIR=${CODE_DIR}

RUN yum update -y && yum install -y git && \
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
    yum update -y && yum install -y \
        unzip \
        wget && \
    wget --quiet -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip && \
    unzip -o protobuf.zip && \
    rm protobuf.zip && \
    ./bin/protoc object_detection/protos/*.proto --python_out=.

RUN yum update -y && yum install -y \
       mesa-libGL \
       glib2-devel

RUN pip install opencv-python

RUN pip install tensorflow-addons==0.11.0

ARG PACKAGE_DIR=model_packages

ARG PACKAGE_NAME="tf-spr-ssd-resnet34-inference"

ARG MODEL_WORKSPACE

# ${MODEL_WORKSPACE} and below needs to be owned by root:root rather than the current UID:GID
# this allows the default user (root) to work in k8s single-node, multi-node
RUN umask 002 && mkdir -p ${MODEL_WORKSPACE} && chgrp root ${MODEL_WORKSPACE} && chmod g+s+w,o+s+r ${MODEL_WORKSPACE}

ADD --chown=0:0 ${PACKAGE_DIR}/${PACKAGE_NAME}.tar.gz ${MODEL_WORKSPACE}

RUN chown -R root ${MODEL_WORKSPACE}/${PACKAGE_NAME} && chgrp -R root ${MODEL_WORKSPACE}/${PACKAGE_NAME} && chmod -R g+s+w ${MODEL_WORKSPACE}/${PACKAGE_NAME} && find ${MODEL_WORKSPACE}/${PACKAGE_NAME} -type d | xargs chmod o+r+x 

WORKDIR ${MODEL_WORKSPACE}/${PACKAGE_NAME}

ENV USER_ID=0

ENV USER_NAME=root

ENV GROUP_ID=0

ENV GROUP_NAME=root

ENV GOSU_VERSION=1.11
RUN curl -o /tmp/gosu.key -SL "https://keys.openpgp.org/vks/v1/by-fingerprint/B42F6819007F00F88E364FD4036A9C25BF357DD4" \
    && gpg --import /tmp/gosu.key \
    && curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/${GOSU_VERSION}/gosu-amd64" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/${GOSU_VERSION}/gosu-amd64.asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && rm -r /root/.gnupg/ \
    && rm /tmp/gosu.key \
    && chmod +x /usr/local/bin/gosu

RUN echo -e '#!/bin/bash\n\
USER_ID=$USER_ID\n\
USER_NAME=$USER_NAME\n\
GROUP_ID=$GROUP_ID\n\
GROUP_NAME=$GROUP_NAME\n\
if [[ $GROUP_NAME != root ]]; then\n\
  groupadd -r -g $GROUP_ID $GROUP_NAME\n\
fi\n\
if [[ $USER_NAME != root ]]; then\n\
  useradd --no-log-init -r -u $USER_ID -g $GROUP_NAME -s /bin/bash -M $USER_NAME\n\
fi\n\
exec /usr/local/bin/gosu $USER_NAME:$GROUP_NAME "$@"\n '\
>> /tmp/entrypoint.sh

RUN chmod u+x,g+x /tmp/entrypoint.sh

ENTRYPOINT ["/tmp/entrypoint.sh"]

ARG TF_BENCHMARKS_BRANCH="509b9d288937216ca7069f31cfb22aaa7db6a4a7"

ARG TF_BENCHMARKS_DIR="/workspace/ssd-resnet-benchmarks"

ENV TF_BENCHMARKS_DIR=${TF_BENCHMARKS_DIR}

RUN yum update -y && yum install -y git && \
    git clone --single-branch https://github.com/tensorflow/benchmarks.git ${TF_BENCHMARKS_DIR} && \
    ( cd ${TF_BENCHMARKS_DIR} && \
    git checkout ${TF_BENCHMARKS_BRANCH} )
