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

ARG PYT_BASE_IMAGE="intel/intel-extension-for-pytorch"
ARG PYT_BASE_TAG="2.0.0-pip-base"

FROM ${PYT_BASE_IMAGE}:${PYT_BASE_TAG} AS intel-extension-for-pytorch
RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    build-essential \
    ca-certificates \
    git \
    wget \
    make \
    cmake \
    g++ \
    gcc \
    autoconf \
    bzip2 \
    tar

RUN wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz && \
    tar -xzf gperftools-2.7.90.tar.gz && \
    cd gperftools-2.7.90 && \
    mkdir -p /workspace/lib/ && \
    ./configure --prefix=/workspace/lib/tcmalloc/ && \
    make && \
    make install

WORKDIR /workspace/pytorch-spr-bert-large-inference
COPY models/language_modeling/pytorch/common/enable_ipex_for_transformers.diff models/language_modeling/pytorch/common/enable_ipex_for_transformers.diff
COPY quickstart/language_modeling/pytorch/bert_large/inference/cpu/configure.json quickstart/configure.json
COPY quickstart/language_modeling/pytorch/bert_large/inference/cpu/run_accuracy.sh quickstart/run_accuracy.sh
COPY quickstart/language_modeling/pytorch/bert_large/inference/cpu/run_calibration.sh quickstart/run_calibration.sh
COPY quickstart/language_modeling/pytorch/bert_large/inference/cpu/run_multi_instance_realtime.sh quickstart/run_multi_instance_realtime.sh
COPY quickstart/language_modeling/pytorch/bert_large/inference/cpu/run_multi_instance_throughput.sh quickstart/run_multi_instance_throughput.sh

RUN cd quickstart && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git checkout v4.28.1 && \
    git apply /workspace/pytorch-spr-bert-large-inference/models/language_modeling/pytorch/common/enable_ipex_for_transformers.diff && \
    pip install -e ./ && \
    pip install tensorboard && \
    pip install intel-openmp 

ENV DNNL_MAX_CPU_ISA="AVX512_CORE_AMX"

# ENV LD_PRELOAD="/workspace/lib/tcmalloc/lib/libtcmalloc.so:/root/conda/envs/pytorch/lib/libiomp5.so:$LD_PRELOAD"
ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    numactl \
    libegl1-mesa

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party
