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

RUN git clone https://github.com/jemalloc/jemalloc.git && \
    cd jemalloc && \
    git checkout c8209150f9d219a137412b06431c9d52839c7272 && \
    ./autogen.sh && \
    ./configure --prefix=/workspace/lib/tcmalloc && \
    make && \ 
    make install

ENV LD_PRELOAD="/workspace/lib/tcmalloc/lib/libjemalloc.so":$LD_PRELOAD

RUN wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz && \
    tar -xzf gperftools-2.7.90.tar.gz && \
    cd gperftools-2.7.90 && \
    mkdir -p /workspace/lib/ && \
    ./configure --prefix=/workspace/lib/tcmalloc/ && \
    make && \
    make install

WORKDIR /workspace/pytorch-spr-distilbert-inference

COPY quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/convert.py quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/convert.py
COPY quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/configure.json configure.json
COPY quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/run_accuracy.sh  quickstart/run_accuracy.sh
COPY quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/do_calibration.sh quickstart/do_calibration.sh
COPY quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/run_multi_instance_realtime.sh quickstart/run_multi_instance_realtime.sh
COPY quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/run_multi_instance_throughput.sh quickstart/run_multi_instance_throughput.sh
COPY quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/download_model_dataset.sh quickstart/download_model_dataset.sh

RUN cd quickstart && \
    wget https://raw.githubusercontent.com/huggingface/datasets/2.0.0/metrics/accuracy/accuracy.py && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git checkout v4.28.1 && \
    pip install -e ./ && \
    pip install -r examples/pytorch/language-modeling/requirements.txt && \
    pip install tensorboard intel-openmp && \
    mkdir -p /root/.local && \
    pip install datasets scikit-learn scipy && \
    apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    git-lfs && \
    git lfs install

ENV DNNL_MAX_CPU_ISA="AVX512_CORE_AMX"

# ENV LD_PRELOAD="/workspace/lib/tcmalloc/lib/libtcmalloc.so:/root/conda/envs/pytorch/lib/libiomp5.so:$LD_PRELOAD"
ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    numactl \
    libegl1-mesa 

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party
