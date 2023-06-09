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

ARG BASE_IMAGE="intel/intel-extension-for-tensorflow"
ARG BASE_TAG="gpu"

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace/tf-flex-series-resnet50v1-5-inference

ARG MODEL_URL

RUN mkdir -p pretrained_models && \ 
    wget ${MODEL_URL} -O pretrained_models/resnet50v1_5-frozen_graph-int8-gpu.pb
    
    
COPY benchmarks/common benchmarks/common
COPY benchmarks/image_recognition/__init__.py benchmarks/image_recognition/__init__.py
COPY benchmarks/image_recognition/tensorflow/__init__.py benchmarks/image_recognition/tensorflow/__init__.py
COPY benchmarks/image_recognition/tensorflow/resnet50v1_5/__init__.py benchmarks/image_recognition/tensorflow/resnet50v1_5/__init__.py
COPY benchmarks/image_recognition/tensorflow/resnet50v1_5/inference benchmarks/image_recognition/tensorflow/resnet50v1_5/inference
COPY benchmarks/launch_benchmark.py benchmarks/launch_benchmark.py
COPY models/common models/common
COPY models/image_recognition/tensorflow/resnet50v1_5 models/image_recognition/tensorflow/resnet50v1_5
COPY quickstart/common quickstart/common
COPY quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/README_Flex_series.md README.md
COPY quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/batch_inference.sh quickstart/batch_inference.sh
COPY quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/online_inference.sh  quickstart/online_inference.sh
COPY quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/accuracy.sh quickstart/accuracy.sh

COPY LICENSE license/LICENSE
COPY third_party license/third_party
