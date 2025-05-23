#!/bin/bash

#
# Copyright (c) 2025 Intel Corporation
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
#


# Clone the Gaudi Tutorial repo in the LLAMA inference directory
cd ${MODEL_DIR}
git clone https://github.com/HabanaAI/Gaudi-tutorials.git
docker build --no-cache -t gaudi-benchmark:latest --build-arg https_proxy=$https_proxy \
--build-arg http_proxy=$http_proxy -f ../../../../../docker/pytorch/llama/inference/hpu/pytorch-gaudi-benchmark.Dockerfile-ubuntu .
