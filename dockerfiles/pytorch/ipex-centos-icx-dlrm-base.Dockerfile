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

ARG BASE_IMAGE=centos:centos7
FROM ${BASE_IMAGE} AS dev-base
SHELL ["/bin/bash", "-c"]
WORKDIR /workspace/installs/



RUN echo "http_caching=packages" >> /etc/yum.conf && \
    yum update -y && \
    yum --enablerepo=extras install -y epel-release && \
    yum install -y \
    ca-certificates \
    git \
    wget \
    build-essential \
    cmake3 \
    gcc-c++ \
    gcc \
    autoconf \
    bzip2 \
    patch \
    numactl \
    file && \
    alternatives --install /usr/local/bin/cmake cmake /usr/bin/cmake 10 \
    --slave /usr/local/bin/ctest ctest /usr/bin/ctest \
    --slave /usr/local/bin/cpack cpack /usr/bin/cpack \
    --slave /usr/local/bin/ccmake ccmake /usr/bin/ccmake \
    --family cmake && \
    alternatives --install /usr/local/bin/cmake cmake /usr/bin/cmake3 20 \
    --slave /usr/local/bin/ctest ctest /usr/bin/ctest3 \
    --slave /usr/local/bin/cpack cpack /usr/bin/cpack3 \
    --slave /usr/local/bin/ccmake ccmake /usr/bin/ccmake3 \
    --family cmake && \
    yum install -y centos-release-scl && \
    yum install -y devtoolset-7 && \
    source /opt/rh/devtoolset-7/enable && \
    wget --quiet https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz && \
    tar -xzf protobuf-2.6.1.tar.gz && \
    cd protobuf-2.6.1 && \
    ./configure && \
    make && \
    make install

RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh && \
    chmod +x anaconda3.sh && \
    ./anaconda3.sh -b -p ~/anaconda3 && \
    rm ./anaconda3.sh && \
    ~/anaconda3/bin/conda create -yn pytorch && \
    export PATH=~/anaconda3/bin/:$PATH && \
    source activate pytorch && \
    pip install pip==21.0.1 && \
    pip install sklearn onnx && \
    conda config --add channels intel && \
    conda install -y ninja pyyaml setuptools cmake cffi typing intel-openmp && \
    conda install -y mkl mkl-include numpy -c intel --no-update-deps

ENV PATH ~/anaconda3/bin/:$PATH

SHELL [ "/usr/bin/scl", "enable", "devtoolset-7"]
ENV DNNL_ENABLE_MAX_CPU_ISA=1
RUN source ~/anaconda3/bin/activate pytorch && \ 
    git clone https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git checkout v1.5.0-rc3 && \
    git submodule sync && \
    git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/intel/intel-extension-for-pytorch ipex-cpu-dev && \
    cd ipex-cpu-dev && \
    git checkout icx && \
    git submodule sync && \
    git submodule update --init --recursive && \
    cd third_party/mkl-dnn && \
    patch -p1 < ../../torch_patches/FIFO.diff && \
    cd ../.. && \
    pip install -r requirements.txt && \
    cp torch_patches/dpcpp-v1.5-rc3.patch ../pytorch/ && \
    cd ../pytorch && \
    git apply dpcpp-v1.5-rc3.patch && \
    CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} python setup.py install && \
    cd ../ipex-cpu-dev && \
    python setup.py install && \
    cd .. && \
    git clone  https://github.com/jemalloc/jemalloc.git && \
    cd jemalloc && \
    git checkout c8209150f9d219a137412b06431c9d52839c7272 && \
    ./autogen.sh && \
    mkdir /workspace/libs && \
    ./configure --prefix=/workspace/libs/jemalloc && \
    make && \
    make install && \
    cd /workspace && \
    rm -rf installs/ && \
    pip install future \
    numpy \
    onnx \
    pydot \
    torchviz \
    scikit-learn \
    tqdm
 

FROM ${BASE_IMAGE} AS release
COPY --from=dev-base /root/anaconda3 /root/anaconda3
COPY --from=dev-base /workspace/libs/ /workspace/libs/

ENV PATH="~/anaconda3/bin:${PATH}"
ENV LD_PRELOAD="/workspace/libs/jemalloc/lib/libjemalloc.so:$LD_PRELOAD"
WORKDIR /workspace/
RUN echo "http_caching=packages" >> /etc/yum.conf && \
    yum update -y --disablerepo=epel\* && \
    yum install -y numactl && \
    yum clean all
