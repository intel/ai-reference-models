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

RUN pip install matplotlib Pillow pycocotools pytz==2018.5 && \
    pip install yacs opencv-python cityscapesscripts transformers defusedxml

WORKDIR /workspace/pytorch-spr-ssd-resnet34-training

COPY models/object_detection/pytorch/ssd-resnet34/training/cpu models/object_detection/pytorch/ssd-resnet34/training/cpu
COPY quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/throughput.sh quickstart/throughput.sh
COPY quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/accuracy.sh quickstart/accuracy.sh

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    libpng-dev \
    libfreetype-dev && \
    pip install --upgrade pip && \
    pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip 

ENV DNNL_MAX_CPU_ISA="AVX512_CORE_AMX"

ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    numactl \
    libegl1-mesa 

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party
