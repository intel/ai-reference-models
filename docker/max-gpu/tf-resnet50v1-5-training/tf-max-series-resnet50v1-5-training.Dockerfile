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
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG TF_BASE_IMAGE="intel/intel-extension-for-tensorflow"
ARG TF_BASE_TAG="2.14.0.1-xpu"

FROM ${TF_BASE_IMAGE}:${TF_BASE_TAG}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl bc

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    intel-oneapi-mpi-devel=2021.11.0-49493  \
    intel-oneapi-ccl=2021.11.2-5 \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/tf-max-series-resnet50v1-5-training/models

COPY models_v2/tensorflow/resnet50v1_5/training/gpu .

RUN python -m pip install gin gin-config \
    tfa-nightly \
    tensorflow-model-optimization \
    tensorflow-datasets \
    protobuf==3.20.3 \
    pyyaml 

RUN mkdir -p resnet50 && \
    cd resnet50 && \
    git clone -b v2.8.0 https://github.com/tensorflow/models.git tensorflow-models && \
    cd tensorflow-models && \
    git apply /workspace/tf-max-series-resnet50v1-5-training/models/resnet50.patch 

RUN mkdir -p resnet50_hvd && \
    cd resnet50_hvd && \
    git clone -b v2.8.0 https://github.com/tensorflow/models.git tensorflow-models && \
    cd tensorflow-models && \
    git apply /workspace/tf-max-series-resnet50v1-5-training/models/hvd_support.patch

ENV LD_LIBRARY_PATH=/opt/intel/oneapi/mpi/2021.11/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.11/lib:/opt/intel/oneapi/ccl/2021.11/lib/:$LD_LIBRARY_PATH
ENV PATH=/opt/intel/oneapi/mpi/2021.11/opt/mpi/libfabric/bin:/opt/intel/oneapi/mpi/2021.11/bin:$PATH
ENV CCL_ROOT=/opt/intel/oneapi/ccl/2021.11
ENV I_MPI_ROOT=/opt/intel/oneapi/mpi/2021.11
ENV FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.11/opt/mpi/libfabric/lib/prov:/usr/lib/x86_64-linux-gnu/libfabric

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party 
