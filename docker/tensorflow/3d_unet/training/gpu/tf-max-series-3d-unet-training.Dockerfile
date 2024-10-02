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

ARG TF_BASE_IMAGE="intel/intel-extension-for-tensorflow"
ARG TF_BASE_TAG="2.15.0.1-xpu"

FROM ${TF_BASE_IMAGE}:${TF_BASE_TAG}

ENV DEBIAN_FRONTEND=noninteractive
    
ARG MPI_VER
ARG CCL_VER

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        intel-oneapi-mpi-devel=${MPI_VER} \
        intel-oneapi-ccl=${CCL_VER} && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/tf-max-series-3d-unet-training/models

RUN python -m pip install tfa-nightly \
        git+https://github.com/NVIDIA/dllogger.git 

COPY models_v2/tensorflow/3d_unet/training/gpu .

RUN mkdir -p 3d_unet && \
    cd 3d_unet && \
    git clone https://github.com/NVIDIA/DeepLearningExamples.git && \
    cd DeepLearningExamples && \
    git checkout 88eb3cff2f03dad85035621d041e23a14345999e && \
    cd TensorFlow/Segmentation/UNet_3D_Medical && \
    git apply /workspace/tf-max-series-3d-unet-training/models/3dunet_itex.patch

RUN mkdir -p 3d_unet_hvd && \
    cd 3d_unet_hvd && \
    git clone https://github.com/NVIDIA/DeepLearningExamples.git && \
    cd DeepLearningExamples && \
    git checkout 88eb3cff2f03dad85035621d041e23a14345999e && \
    cd TensorFlow/Segmentation/UNet_3D_Medical && \
    git apply /workspace/tf-max-series-3d-unet-training/models/3dunet_itex_with_horovod.patch

ENV LD_LIBRARY_PATH=/opt/intel/oneapi/mpi/2021.13/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.13/lib:/opt/intel/oneapi/ccl/2021.13/lib/:$LD_LIBRARY_PATH
ENV PATH=/opt/intel/oneapi/mpi/2021.13/opt/mpi/libfabric/bin:/opt/intel/oneapi/mpi/2021.13/bin:$PATH
ENV CCL_ROOT=/opt/intel/oneapi/ccl/2021.13
ENV I_MPI_ROOT=/opt/intel/oneapi/mpi/2021.13
ENV FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.13/opt/mpi/libfabric/lib/prov:/usr/lib/x86_64-linux-gnu/libfabric

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party 
