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

ARG TF_BASE_IMAGE="intel/intel-extension-for-tensorflow"
ARG TF_BASE_TAG="xpu"

FROM ${TF_BASE_IMAGE}:${TF_BASE_TAG}

WORKDIR /workspace/tf-flex-series-resnet50v1-5-inference/models

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing parallel pciutils numactl

ARG MODEL_URL

RUN mkdir -p pretrained_models && \ 
    wget ${MODEL_URL} -O pretrained_models/resnet50v1_5-frozen_graph-int8-gpu.pb
    
COPY models_v2/tensorflow/resnet50v1_5/inference/gpu .

COPY LICENSE license/LICENSE
COPY third_party license/third_party
