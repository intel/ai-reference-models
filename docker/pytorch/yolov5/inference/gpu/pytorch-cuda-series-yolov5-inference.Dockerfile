# Copyright (c) 2023-2024 Intel Corporation
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

ARG CUDA_BASE_IMAGE="nvcr.io/nvidia/pytorch"
ARG CUDA_BASE_TAG="24.02-py3"

FROM ${CUDA_BASE_IMAGE}:${CUDA_BASE_TAG}

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

ARG WORKDIR=/workspace/pytorch-cuda-series-yolov5-inference

WORKDIR $WORKDIR
COPY models_v2/pytorch/yolov5/inference/gpu .
COPY models_v2/common common

ENV PYTHONPATH=$WORKDIR/common

RUN apt-get update && \
    apt-get install -y --no-install-recommends pciutils numactl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Required to fix "fatal: module 'cv2.dnn' has no attribute 'DictValue'" in CUDA container
# See: https://github.com/opencv/opencv-python/issues/884
RUN pip install opencv-fixer==0.2.5
RUN python3 -c "from opencv_fixer import AutoFix; AutoFix()"

COPY LICENSE license/LICENSE
COPY third_party license/third_party
