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

ARG PYTORCH_IMAGE="model-zoo"
ARG PYTORCH_TAG="pytorch-ipex-spr"

FROM ${PYTORCH_IMAGE}:${PYTORCH_TAG} AS intel-optimized-pytorch

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
    tar

# Build Torch Vision
ARG TORCHVISION_VERSION="81fe60cc258f49ccfb0f9f32d78b4825754cff7b"

RUN source activate pytorch && \
    git clone https://github.com/pytorch/vision && \
    cd vision && \
    git checkout ${TORCHVISION_VERSION} && \
    python setup.py install

RUN source activate pytorch && \
    pip install matplotlib Pillow pycocotools && \
    pip install yacs opencv-python cityscapesscripts transformers && \
    conda install -y libopenblas && \
    mkdir -p /workspace/installs && \
    cd /workspace/installs && \
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz && \
    tar -xzf gperftools-2.7.90.tar.gz && \
    cd gperftools-2.7.90 && \
    ./configure --prefix=$HOME/.local && \
    make && \
    make install && \
    rm -rf /workspace/installs/

ARG PACKAGE_DIR=model_packages

ARG PACKAGE_NAME="pytorch-spr-maskrcnn-training"

ARG MODEL_WORKSPACE

# ${MODEL_WORKSPACE} and below needs to be owned by root:root rather than the current UID:GID
# this allows the default user (root) to work in k8s single-node, multi-node
RUN umask 002 && mkdir -p ${MODEL_WORKSPACE} && chgrp root ${MODEL_WORKSPACE} && chmod g+s+w,o+s+r ${MODEL_WORKSPACE}

ADD --chown=0:0 ${PACKAGE_DIR}/${PACKAGE_NAME}.tar.gz ${MODEL_WORKSPACE}

RUN chown -R root ${MODEL_WORKSPACE}/${PACKAGE_NAME} && chgrp -R root ${MODEL_WORKSPACE}/${PACKAGE_NAME} && chmod -R g+s+w ${MODEL_WORKSPACE}/${PACKAGE_NAME} && find ${MODEL_WORKSPACE}/${PACKAGE_NAME} -type d | xargs chmod o+r+x 

WORKDIR ${MODEL_WORKSPACE}/${PACKAGE_NAME}

RUN source activate pytorch && \
    cd ${MODEL_WORKSPACE}/${PACKAGE_NAME}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark && \
    python setup.py install && \
    pip install onnx

FROM intel-optimized-pytorch AS release
COPY --from=intel-optimized-pytorch /root/conda /root/conda
COPY --from=intel-optimized-pytorch /workspace/lib/ /workspace/lib/
COPY --from=intel-optimized-pytorch /root/.local/ /root/.local/

ENV DNNL_MAX_CPU_ISA="AVX512_CORE_AMX"

ENV PATH="~/conda/bin:${PATH}"
ENV LD_PRELOAD="/workspace/lib/jemalloc/lib/libjemalloc.so:/root/conda/envs/pytorch/lib/libiomp5.so:$LD_PRELOAD"
ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
ENV BASH_ENV=/root/.bash_profile
WORKDIR /workspace/
RUN yum install -y numactl mesa-libGL && \
    yum clean all && \
    echo "export LD_PRELOAD=${LD_PRELOAD%%:}" >> /root/.bash_profile && \
    echo "source activate pytorch" >> /root/.bash_profile
