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

RUN yum update -y && \
    yum install -y gcc gcc-c++ cmake python3-tkinter libXext libSM && \
    yum clean all

# Install OpenMPI
ARG OPENMPI_VERSION="openmpi-4.1.0"
ARG OPENMPI_DOWNLOAD_URL="https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.0.tar.gz"

RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    curl -fSsL -O ${OPENMPI_DOWNLOAD_URL} && \
    tar zxf ${OPENMPI_VERSION}.tar.gz && \
    cd ${OPENMPI_VERSION} && \
    ./configure --enable-mpirun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/openmpi

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
    chmod a+x /usr/local/bin/mpirun

# Configure OpenMPI to run good defaults:
RUN echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf

# Install OpenSSH for MPI to communicate between containers
RUN yum update -y && yum install -y  \
    openssh-server \
    openssh-clients && \
    yum clean all

ARG HOROVOD_VERSION=11c1389
ENV HOROVOD_WITHOUT_MXNET=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_TENSORFLOW=1 \
    HOROVOD_CPU_OPERATIONS=MPI \
    HOROVOD_WITH_MPI=1 \
    HOROVOD_WITHOUT_GLOO=1

# Install Horovod
RUN yum update -y && yum install -y git cmake gcc-c++ && \
    yum clean all

RUN python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}

RUN pip install opencv-python

RUN yum update -y && yum install -y numactl

RUN pip install tensorflow-addons==0.11.0

ARG TF_MODELS_BRANCH="8110bb64ca63c48d0caee9d565e5b4274db2220a"

ARG FETCH_PR

ARG CODE_DIR=/tensorflow/models

ENV TF_MODELS_DIR=${CODE_DIR}

RUN yum update -y && yum install -y git && \
    git clone https://github.com/tensorflow/models.git ${CODE_DIR} && \
    ( cd ${CODE_DIR} && \
    if [ ! -z "${FETCH_PR}" ]; then git fetch origin ${FETCH_PR}; fi && \
    git checkout ${TF_MODELS_BRANCH} )

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

ARG PACKAGE_DIR=model_packages

ARG PACKAGE_NAME="tf-spr-ssd-resnet34-training"

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
