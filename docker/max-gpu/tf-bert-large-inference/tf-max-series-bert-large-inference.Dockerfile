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

ARG BASE_IMAGE="intel/intel-extension-for-tensorflow"
ARG BASE_TAG="xpu"

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace/tf-max-series-bert-large-inference

ARG MODEL_URL

RUN mkdir -p frozen_graph && \
    wget ${MODEL_URL} -O frozen_graph/fp32_bert_squad.pb

COPY models/language_modeling/tensorflow/bert_large/inference/gpu models/language_modeling/tensorflow/bert_large/inference/gpu

COPY quickstart/language_modeling/tensorflow/bert_large/inference/gpu/inference.sh quickstart/inference.sh
COPY quickstart/language_modeling/tensorflow/bert_large/inference/gpu/README.md quickstart/README.md

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party 
