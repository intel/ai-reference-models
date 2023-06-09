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

ARG MAX_PYT_BASE_IMAGE="intel/intel-extension-for-pytorch"
ARG MAX_PYT_BASE_TAG="xpu-max"

FROM ${MAX_PYT_BASE_IMAGE}:${MAX_PYT_BASE_TAG}

WORKDIR /workspace/pytorch-max-series-bert-large-inference

COPY quickstart/language_modeling/pytorch/bert_large/inference/gpu/README.md README.md
COPY models/language_modeling/pytorch/bert_large/inference/gpu models/language_modeling/pytorch/bert_large/inference/gpu

RUN cd /workspace/pytorch-max-series-bert-large-inference/models/language_modeling/pytorch/bert_large/inference/gpu && \
    pip install -r requirements.txt 

COPY quickstart/language_modeling/pytorch/bert_large/inference/gpu/fp16_inference_plain_format.sh quickstart/fp16_inference_plain_format.sh

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party
