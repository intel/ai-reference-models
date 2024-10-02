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

ARG PYT_BASE_IMAGE="intel/intel-extension-for-pytorch"
ARG PYT_BASE_TAG="2.3.110-xpu"

FROM ${PYT_BASE_IMAGE}:${PYT_BASE_TAG}

WORKDIR /workspace/pytorch-max-series-rnnt-training/models

ENV DEBIAN_FRONTEND=noninteractive

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
   | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
   | tee /etc/apt/sources.list.d/oneAPI.list

ARG MPI_VER
ARG CCL_VER

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        intel-oneapi-mpi-devel=${MPI_VER} \
        intel-oneapi-ccl=${CCL_VER} && \
    rm -rf /var/lib/apt/lists/*
    
COPY models_v2/pytorch/rnnt/training/gpu .
COPY models_v2/common common

RUN python -m pip install -r requirements.txt 

RUN python -m pip install --no-cache-dir --upgrade pip

ENV LD_LIBRARY_PATH=/opt/intel/oneapi/ccl/2021.13/lib/:/opt/intel/oneapi/mpi/2021.13/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.13/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/intel/oneapi/mpi/2021.13/lib:/opt/intel/oneapi/ccl/2021.13/lib/
ENV PATH=/opt/intel/oneapi/mpi/2021.13/opt/mpi/libfabric/bin:/opt/intel/oneapi/mpi/2021.13/bin:$PATH
ENV CCL_ROOT=/opt/intel/oneapi/ccl/2021.13
ENV I_MPI_ROOT=/opt/intel/oneapi/mpi/2021.13
ENV FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.13/opt/mpi/libfabric/lib/prov:/usr/lib/x86_64-linux-gnu/libfabric

COPY LICENSE license/LICENSE
COPY third_party license/third_party
