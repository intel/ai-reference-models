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

ARG BASE_IMAGE="intel/intel-extension-for-pytorch"
ARG BASE_TAG="xpu-max"

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace/pytorch-max-series-dlrm-training

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing numactl 
    
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | apt-key add -
RUN echo "deb [trusted=yes] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    intel-oneapi-mpi-devel=2021.10.0-49371 \
    intel-oneapi-ccl=2021.10.0-49084 \
    && \
    rm -rf /var/lib/apt/lists/*

COPY models/recommendation/pytorch/torchrec_dlrm/training/gpu models/recommendation/pytorch/torchrec_dlrm/training/gpu
COPY quickstart/recommendation/pytorch/torchrec_dlrm/training/gpu/multi_card_distributed_train.sh quickstart/multi_card_distributed_train.sh 

RUN cd models/recommendation/pytorch/torchrec_dlrm/training/gpu && \
    pip install -r requirements.txt && \
    cd -
RUN pip install -e git+https://github.com/mlperf/logging#egg=mlperf-logging

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party
