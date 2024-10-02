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

WORKDIR /workspace/tf-flex-series-stable-diffusion-inference/models

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY models_v2/tensorflow/stable_diffusion/inference/gpu/ .


RUN git clone https://github.com/keras-team/keras-cv.git && \
    cd keras-cv && \
    git reset --hard 66fa74b6a2a0bb1e563ae8bce66496b118b95200 && \
    mv /workspace/tf-flex-series-stable-diffusion-inference/models/patch . && \
    git apply patch && \
    pip install matplotlib && \
    pip install .

RUN python -m pip install scikit-image scipy==1.11.1

COPY LICENSE license/LICENSE
COPY third_party license/third_party
