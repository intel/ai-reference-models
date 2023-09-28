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


ARG BASE_IMAGE="intel/intel-extension-for-pytorch"
ARG BASE_TAG="2.0.0-pip-base"

FROM ${BASE_IMAGE}:${BASE_TAG} AS intel-optimized-pytorch

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    build-essential \
    ca-certificates \
    git \
    wget \
    make \
    cmake \
    autoconf \
    bzip2 \
    tar

# Build Torch Vision
ARG TORCHVISION_VERSION="01c11a0564b8417561ae4c414fe659fc97476987"

RUN git clone https://github.com/pytorch/vision && \
    cd vision && \
    git checkout ${TORCHVISION_VERSION} && \
    python setup.py install

RUN git clone https://github.com/jemalloc/jemalloc.git && \
    cd jemalloc && \
    git checkout c8209150f9d219a137412b06431c9d52839c7272 && \
    ./autogen.sh && \
    ./configure --prefix=/workspace/lib/tcmalloc && \
    make && \ 
    make install

ENV LD_PRELOAD="/workspace/lib/tcmalloc/lib/libjemalloc.so":$LD_PRELOAD

WORKDIR /workspace/pytorch-spr-rnnt-training 

COPY models/language_modeling/pytorch/rnnt/training/cpu models/language_modeling/pytorch/rnnt/training/cpu
COPY quickstart/language_modeling/pytorch/rnnt/training/cpu/training.sh quickstart/training.sh
COPY quickstart/language_modeling/pytorch/rnnt/training/cpu/download_dataset.sh quickstart/download_dataset.sh

RUN cd /workspace/pytorch-spr-rnnt-training/models/language_modeling/pytorch/rnnt/training/cpu && \
    pip install -r requirements.txt && \
    pip install unidecode inflect && \
    pip install --force-reinstall soundfile && \
    pip install protobuf==3.20.* && \
    pip install --upgrade pip && \
    pip install librosa sox && \
    apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    libsndfile1 && \
    mkdir -p /root/.local && \
    git clone https://github.com/HawkAaron/warp-transducer && \
    cd warp-transducer && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cd ../pytorch_binding && \
    apt-get update && apt-get install -y python3.10-dev && \
    python setup.py install

ENV DNNL_MAX_CPU_ISA="AVX512_CORE_AMX"

# ENV LD_PRELOAD="/workspace/lib/jemalloc/lib/libjemalloc.so:/root/conda/envs/pytorch/lib/libiomp5.so:$LD_PRELOAD"
ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    numactl \
    libegl1-mesa 

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party
