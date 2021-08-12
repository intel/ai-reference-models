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

ARG BASE_IMAGE=centos:centos8.3.2011

FROM ${BASE_IMAGE} AS centos-intel-base
SHELL ["/bin/bash", "-c"]

RUN yum update -y && yum install -y unzip

FROM centos-intel-base as ipex-dev-base
WORKDIR /workspace/installs/
RUN yum --enablerepo=extras install -y epel-release && \
    yum install -y \
    ca-certificates \
    git \
    wget \
    make \
    cmake \
    gcc-c++ \
    gcc \
    autoconf \
    bzip2 \
    numactl \
    nc \
    tar \
    patch && \
    wget --quiet https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz && \
    tar -xzf protobuf-2.6.1.tar.gz && \
    cd protobuf-2.6.1 && \
    ./configure && \
    make && \
    make install

# Prepare the Conda environment
RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh && \
    chmod +x anaconda3.sh && \
    ./anaconda3.sh -b -p ~/anaconda3 && \
    rm ./anaconda3.sh && \
    ~/anaconda3/bin/conda create -yn pytorch python=3.7 && \
    export PATH=~/anaconda3/bin/:${PATH} && \
    source activate pytorch && \
    pip install pip==21.0.1 && \
    pip install sklearn onnx && \
    conda config --add channels intel && \
    conda install -y ninja pyyaml setuptools cmake cffi typing intel-openmp && \
    conda install -y mkl mkl-include numpy -c intel --no-update-deps

ENV PATH ~/anaconda3/bin/:${PATH}
ENV LD_LIBRARY_PATH /lib64/:/usr/lib64/:/usr/local/lib64:/root/anaconda3/envs/pytorch/lib:${LD_LIBRARY_PATH}

# Install PyTorch and IPEX wheels
ARG PYTORCH_WHEEL
ARG IPEX_WHEEL

COPY ./whls/* /tmp/pip3/
RUN source ~/anaconda3/bin/activate pytorch && \
    pip install /tmp/pip3/${PYTORCH_WHEEL} && \
    pip install /tmp/pip3/${IPEX_WHEEL}

# Build Jemalloc
ARG JEMALLOC_SHA=c8209150f9d219a137412b06431c9d52839c7272

RUN source ~/anaconda3/bin/activate pytorch && \
    git clone  https://github.com/jemalloc/jemalloc.git && \
    cd jemalloc && \
    git checkout ${JEMALLOC_SHA} && \
    ./autogen.sh && \
    mkdir /workspace/lib/ && \
    ./configure --prefix=/workspace/lib/jemalloc/ && \
    make && \
    make install
