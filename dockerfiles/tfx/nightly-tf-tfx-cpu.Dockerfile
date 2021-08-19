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

ARG UBUNTU_VERSION="20.04"

FROM ubuntu:${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get install -y python3.6 && \
    ln -s /usr/bin/python3.6 /usr/bin/python

RUN apt-get update && \
    apt-get install -y python3-pip

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
        libsm6 \
        libxext6 \
        python-tk && \
    pip install requests

ARG PY_VERSION="3.6"

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
        build-essential \
        python${PY_VERSION}-dev

RUN python -m pip install --pre tfx && \
    python -m pip install \
        struct2tensor \
        tensorflow-ranking

RUN python -m pip uninstall -y tensorflow

RUN python -m pip install tf-nightly-cpu

RUN python -m pip install \
        tensorflow-text-nightly \
        tflite-support-nightly \
        pyparsing==2.4.7
