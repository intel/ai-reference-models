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

ARG BASE_IMAGE=centos:centos8.3.2011

FROM ${BASE_IMAGE} AS centos-intel-base
SHELL ["/bin/bash", "-c"]

ENV DNNL_MAX_CPU_ISA="AVX512_CORE_AMX"

# set env var as we moved from block format to native format
ENV TF_ENABLE_MKL_NATIVE_FORMAT=1

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ARG PYTHON=python3

RUN yum update -y && yum install -y \
    ${PYTHON} \
    ${PYTHON}-pip \
    which && \
    yum clean all


RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python

# Installs the latest version by default.
ARG TF_WHEEL=tf_nightly-2.7.0-cp36-cp36m-linux_x86_64.whl

COPY ./whls/${TF_WHEEL} /tmp/pip3/

RUN python3 -m pip install --no-cache-dir /tmp/pip3/${TF_WHEEL}

# fix keras-nightly and tf-estimator-nightly versions
RUN pip uninstall -y keras-nightly tf-estimator-nightly
RUN pip install tf-estimator-nightly==2.7.0.dev2021080801 \
                keras-nightly==2.7.0.dev2021080800
