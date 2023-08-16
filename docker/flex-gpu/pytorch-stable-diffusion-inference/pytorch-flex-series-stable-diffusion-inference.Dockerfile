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

ARG PYTORCH_BASE_IMAGE="intel/intel-extension-for-pytorch"
ARG PYTORCH_BASE_TAG="xpu-flex"

FROM ${PYTORCH_BASE_IMAGE}:${PYTORCH_BASE_TAG}

WORKDIR /home/user/workspace/pytorch-flex-series-stable-diffusion-inference 

RUN apt-get update && \
    apt-get install -y parallel 
RUN apt-get install -y pciutils

ARG PY_VERSION=3.10

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    python${PY_VERSION}-dev

RUN pip install diffusers pytorch-fid transformers

COPY models/generative-ai/pytorch/stable_diffusion/inference/gpu models/generative-ai/pytorch/stable_diffusion/inference/gpu 
COPY quickstart/generative-ai/pytorch/stable_diffusion/inference/gpu/online_inference.sh quickstart/online_inference.sh 

COPY LICENSE license/LICENSE
COPY third_party license/third_party
