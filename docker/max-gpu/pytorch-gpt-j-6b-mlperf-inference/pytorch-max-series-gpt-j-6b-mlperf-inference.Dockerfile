# Copyright (c) 2023 Intel Corporation
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


ARG BASE_IMAGE="intel/oneapi-basekit"
ARG BASE_TAG="2023.1.0-devel-ubuntu22.04"

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace/pytorch-max-series-gpt-j-6b-mlperf-inference
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing cmake libblas-dev \
                       liblapack-dev autoconf unzip \
                       wget git \
                       ca-certificates pkg-config build-essential 

ARG PYTHON
RUN apt-get update && apt install -y software-properties-common 
RUN add-apt-repository -y ppa:deadsnakes/ppa 

RUN apt-cache policy $PYTHON && apt-get update && apt-get install -y \
    --no-install-recommends --fix-missing $PYTHON 

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    ${PYTHON} lib${PYTHON} python3-pip python3.9-dev ${PYTHON}-distutils ${PYTHON}-venv python3-wheel && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/bin/python3

ENV BUILD_SEPARATE_OPS=ON
ENV USE_XETLA=ON
ENV BUILD_WITH_CPU=OFF
ENV USE_AOT_DEVLIST='pvc'

COPY models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu
COPY quickstart/generative-ai/pytorch/gpt-j-6b-mlperf/inference/gpu/inference.sh quickstart/inference.sh

ARG ICD_VER=23.17.26241.33-647~22.04
ARG LEVEL_ZERO_GPU_VER=1.3.26241.33-647~22.04
ARG LEVEL_ZERO_VER=1.11.0-647~22.04
ARG LEVEL_ZERO_DEV_VER=1.11.0-647~22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    intel-opencl-icd=${ICD_VER} \
    intel-level-zero-gpu=${LEVEL_ZERO_GPU_VER} \
    level-zero=${LEVEL_ZERO_VER} \
    level-zero-dev=${LEVEL_ZERO_DEV_VER} && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

RUN wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz && \
    tar -xzf gperftools-2.7.90.tar.gz && \
    cd gperftools-2.7.90 && \
    ./configure --prefix=/workspace/lib/tcmalloc/ && \
    make && \
    make install 

RUN mkdir -p venvs

ARG VENV=/workspace/pytorch-max-series-gpt-j-6b-mlperf-inference/venvs/fp16-benchmark-env/bin

RUN ${PYTHON} -m venv venvs/fp16-benchmark-env && \
    ${VENV}/pip install pybind11==2.11.1 wheel setuptools \
    torch==2.0.1a0 -f https://developer.intel.com/ipex-whl-stable-xpu --proxy ${http_proxy} \
    nltk evaluate rouge_score accelerate==0.21.0 texttable && \
    export PATH=${VENV}:${PATH} && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference && \
    cd mlperf_inference/loadgen && \
    git checkout v3.1 && \
    CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel && \
    cd .. && \
    ${VENV}/pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` && \
    cd ${WORKDIR} && \
    mkdir -p fp16-ipex && \
    cd fp16-ipex && \
    git clone https://github.com/intel/intel-extension-for-pytorch.git && \
    cd intel-extension-for-pytorch && \
    git checkout dev/LLM-MLPerf && \
    git submodule sync &&  git submodule update --init --recursive && \
    ${VENV}/pip install -r requirements.txt && \
    python setup.py install && \
    cd ${WORKDIR} && \
    ${VENV}/pip install transformers==4.29.2 && \
    rm -rf fp16-ipex

ARG VENV=/workspace/pytorch-max-series-gpt-j-6b-mlperf-inference/venvs/int4-quantization-env/bin 

RUN ${PYTHON} -m venv venvs/int4-quantization-env && \
    ${VENV}/pip install pybind11==2.11.1 wheel setuptools  \
    torch==2.0.1a0 -f https://developer.intel.com/ipex-whl-stable-xpu --proxy ${http_proxy} \
    nltk evaluate rouge_score accelerate==0.21.0 texttable && \
    export PATH=${VENV}:${PATH} && \
    cd mlperf_inference/loadgen && \
    CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel && \
    cd .. && \
    ${VENV}/pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` && \
    cd ${WORKDIR} && \
    mkdir -p int4-ipex && \
    cd int4-ipex && \
    git clone https://github.com/intel/intel-extension-for-pytorch.git && \
    cd intel-extension-for-pytorch && \
    git checkout dev/LLM-INT4 && \
    git submodule sync &&  git submodule update --init --recursive && \
    ${VENV}/pip install -r requirements.txt && \
    python setup.py install && \
    cd ${WORKDIR} && \
    ${VENV}/pip install transformers==4.21.2

ARG VENV=/workspace/pytorch-max-series-gpt-j-6b-mlperf-inference/venvs/int4-benchmark-env/bin 

RUN ${PYTHON} -m venv venvs/int4-benchmark-env && \
    ${VENV}/pip install pybind11==2.11.1 wheel setuptools  \
    torch==2.0.1a0 -f https://developer.intel.com/ipex-whl-stable-xpu --proxy ${http_proxy} \
    nltk evaluate rouge_score accelerate==0.21.0 texttable && \  
    export PATH=${VENV}:${PATH} && \
    cd mlperf_inference/loadgen && \
    CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel && \
    cd .. && \
    ${VENV}/pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` && \
    cd ${WORKDIR} && \
    cd int4-ipex/intel-extension-for-pytorch && \
    git checkout dev/LLM-INT4 && \
    git submodule sync &&  git submodule update --init --recursive && \
    ${VENV}/pip  install -r requirements.txt && \
    python setup.py install && \
    cd ${WORKDIR} && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git checkout v4.29.2 && \
    git apply /workspace/pytorch-max-series-gpt-j-6b-mlperf-inference/models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu/patches/int4-transformers.patch && \
    python setup.py install && \
    cd ${WORKDIR} && \
    rm -rf int4-ipex 

RUN apt-get update && apt-get -y upgrade 

RUN apt-get clean && \
    rm -rf  /var/lib/apt/lists/* \
    mlperf_inference \
    gperftools-*
    
COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party
