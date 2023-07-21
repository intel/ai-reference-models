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

ARG MAX_TF_BASE_IMAGE="intel/intel-extension-for-tensorflow"
ARG MAX_TF_BASE_TAG="gpu-horovod"

FROM ${MAX_TF_BASE_IMAGE}:${MAX_TF_BASE_TAG}

WORKDIR /workspace/tf-max-series-resnet50v1-5-inference

ARG INT8_MODEL_URL
ARG FP16_MODEL_URL
ARG FP32_MODEL_URL

RUN mkdir -p pretrained_models && \
    wget ${INT8_MODEL_URL} -O pretrained_models//resnet50v1_5-frozen_graph-int8-gpu.pb && \
    wget ${FP32_MODEL_URL} -O pretrained_models/resnet50v1_5-frozen_graph-fp32-gpu.pb && \
    wget ${FP16_MODEL_URL} -O pretrained_models/resnet50v1_5-frozen_graph-fp16-gpu.pb 

COPY benchmarks benchmarks
COPY models models
COPY quickstart/common quickstart/common

COPY quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/README_Max_Series.md README.md 
COPY quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/accuracy.sh quickstart/accuracy.sh
COPY quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/batch_inference.sh quickstart/batch_inference.sh
COPY quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/online_inference.sh quickstart/online_inference.sh

COPY LICENSE licenses/LICENSE
COPY third_party licenses/third_party 
