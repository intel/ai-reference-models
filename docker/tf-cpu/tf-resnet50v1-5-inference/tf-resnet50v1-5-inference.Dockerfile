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

ARG TF_BASE_IMAGE=intel/intel-optimized-tensorflow-avx512

ARG TF_BASE_TAG=latest

FROM ${TF_BASE_IMAGE}:${TF_BASE_TAG}

WORKDIR /workspace/tf-spr-resnet50v1-5-inference

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y ca-certificates numactl wget

RUN mkdir -p pretrained_model && \
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_8/bias_resnet50.pb -O pretrained_model/bias_resnet50.pb && \
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_8/bf16_resnet50_v1.pb -O pretrained_model/bf16_resnet50_v1.pb && \
    wget https://zenodo.org/record/2535873/files/resnet50_v1.pb -O pretrained_model/resnet50_v1.pb

COPY benchmarks benchmarks
COPY models models
COPY models_v2/common models_v2/common
COPY models_v2/tensorflow/resnet50v1_5/inference/cpu/accuracy.sh models_v2/accuracy.sh
COPY models_v2/tensorflow/resnet50v1_5/inference/cpu/inference_throughput_multi_instance.sh models_v2/inference_throughput.sh
COPY models_v2/tensorflow/resnet50v1_5/inference/cpu/inference_realtime_multi_instance.sh models_v2/inference_realtime.sh
COPY models_v2/tensorflow/resnet50v1_5/inference/cpu/inference_realtime_weightsharing.sh models_v2/inference_realtime_weightsharing.sh

COPY LICENSE license/LICENSE
COPY third_party license/third_party
