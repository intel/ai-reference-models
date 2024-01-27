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

ARG PYT_BASE_IMAGE="intel/intel-extension-for-pytorch"
ARG PYT_BASE_TAG="2.1.10-xpu"

FROM ${PYT_BASE_IMAGE}:${PYT_BASE_TAG}

ENV DEBIAN_FRONTEND=noninteractive
USER root

WORKDIR /workspace/pytorch-flex-series-dlrm-v1-inference/models

RUN python -m pip install future \
                    numpy \
                    onnx \
                    pydot \
                    scikit-learn \
                    tqdm \
                    torchrec-nightly \
                    torchx-nightly \
                    lark-parser \
                    scikit-learn \
                    tensorboard 

COPY models_v2/pytorch/dlrm/inference/gpu .
COPY models_v2/common common

COPY LICENSE license/LICENSE
COPY third_party license/third_party

USER $USER
