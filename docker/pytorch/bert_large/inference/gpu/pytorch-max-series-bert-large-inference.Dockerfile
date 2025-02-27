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

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        parallel && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/pytorch-max-series-bert-large-inference/models

COPY models_v2/pytorch/bert_large/inference/gpu .
COPY models_v2/common/parse_result.py common/parse_result.py 

RUN python -m pip install --no-cache-dir -r requirements.txt 

RUN python -m pip install --no-cache-dir --upgrade pip Pillow==10.2.0 \
        jinja2==3.1.3 
        
COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party
