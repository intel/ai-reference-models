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

ARG BASE_IMAGE="intel/intel-extension-for-tensorflow"
ARG BASE_TAG="xpu"

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace/tf-flex-series-stable-diffusion-inference

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing git

COPY models/generative-ai/tensorflow/stable_diffusion/inference/gpu/ models/generative-ai/tensorflow/stable_diffusion/inference/gpu/

COPY quickstart/generative-ai/tensorflow/stable_diffusion/inference/gpu/online_inference.sh quickstart/online_inference.sh
COPY quickstart/generative-ai/tensorflow/stable_diffusion/inference/gpu/accuracy.sh quickstart/accuracy.sh

RUN git clone https://github.com/keras-team/keras-cv.git && \
    cd keras-cv && \
    git reset --hard 66fa74b6a2a0bb1e563ae8bce66496b118b95200 && \
    mv /workspace/tf-flex-series-stable-diffusion-inference/models/generative-ai/tensorflow/stable_diffusion/inference/gpu/patch . && \
    git apply patch && \
    pip install matplotlib && \
    pip install .

RUN python -m pip install scikit-image

COPY LICENSE license/LICENSE
COPY third_party license/third_party
