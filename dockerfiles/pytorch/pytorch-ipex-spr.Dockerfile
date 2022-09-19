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

ARG BASE_IMAGE=centos:8

FROM ${BASE_IMAGE} AS centos-intel-base
SHELL ["/bin/bash", "-c"]

# Fixe for “Error: Failed to download metadata for repo 'appstream': Cannot prepare internal mirrorlist: No URLs in mirrorlist"
RUN sed -i.bak '/^mirrorlist=/s/mirrorlist=/#mirrorlist=/g' /etc/yum.repos.d/CentOS-Linux-* && \
    sed -i.bak 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-* && \
    yum distro-sync -y && \
    yum --disablerepo '*' --enablerepo=extras swap centos-linux-repos centos-stream-repos -y && \
    yum distro-sync -y && \
    yum clean all

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
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p ~/conda && \
    rm ./miniconda.sh && \
    ~/conda/bin/conda create -yn pytorch python=3.7 && \
    export PATH=~/conda/bin/:${PATH} && \
    source activate pytorch && \
    pip install pip==21.0.1 && \
    conda config --add channels intel && \
    conda install -y ninja pyyaml setuptools cmake cffi typing intel-openmp psutil && \
    conda install -y mkl mkl-include numpy -c intel --no-update-deps

ENV PATH ~/conda/bin/:${PATH}
ENV LD_LIBRARY_PATH /lib64/:/usr/lib64/:/usr/local/lib64:/root/conda/envs/pytorch/lib:${LD_LIBRARY_PATH}

# Install PyTorch and IPEX wheels
ARG PYTORCH_WHEEL="torch-1.12.0a0+gitd2ae05f-cp37-cp37m-linux_x86_64.whl"
ARG IPEX_WHEEL="intel_extension_for_pytorch-1.12.0+cpu-cp37-cp37m-linux_x86_64.whl"

COPY ./whls/* /tmp/pip3/
RUN source activate pytorch && \
    pip install /tmp/pip3/${IPEX_WHEEL} && \
    pip install /tmp/pip3/${PYTORCH_WHEEL}


# Build Jemalloc
ARG JEMALLOC_SHA=c8209150f9d219a137412b06431c9d52839c7272

RUN source activate pytorch && \
    git clone  https://github.com/jemalloc/jemalloc.git && \
    cd jemalloc && \
    git checkout ${JEMALLOC_SHA} && \
    ./autogen.sh && \
    mkdir /workspace/lib/ && \
    ./configure --prefix=/workspace/lib/jemalloc/ && \
    make && \
    make install

FROM centos-intel-base AS release
COPY --from=ipex-dev-base /root/conda /root/conda
COPY --from=ipex-dev-base /workspace/lib /workspace/lib

ENV LD_LIBRARY_PATH /lib64/:/usr/lib64/:/usr/local/lib64:/root/conda/envs/pytorch/lib:${LD_LIBRARY_PATH}
ENV PATH="~/conda/bin:${PATH}"
ENV DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
ENV BASH_ENV=/root/.bash_profile
WORKDIR /workspace/
RUN yum install -y numactl mesa-libGL && \
    yum clean all && \
    echo "source activate pytorch" >> /root/.bash_profile

# Please see: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-0778
RUN yum erase openssl -y && \
    yum clean all
