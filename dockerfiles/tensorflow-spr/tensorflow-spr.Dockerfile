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

ARG BASE_IMAGE=centos:8

FROM ${BASE_IMAGE} AS centos-intel-base
SHELL ["/bin/bash", "-c"]

# Fixe for “Error: Failed to download metadata for repo 'appstream': Cannot prepare internal mirrorlist: No URLs in mirrorlist"
RUN sed -i.bak '/^mirrorlist=/s/mirrorlist=/#mirrorlist=/g' /etc/yum.repos.d/CentOS-Linux-* && \
    sed -i.bak 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-* && \
    yum distro-sync -y && \
    yum --disablerepo '*' --enablerepo=extras swap centos-linux-repos centos-stream-repos -y && \
    yum distro-sync -y && \
    yum clean all

ENV DNNL_MAX_CPU_ISA="AVX512_CORE_AMX"

# set env var as we moved from block format to native format
ENV TF_ENABLE_MKL_NATIVE_FORMAT=1

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ARG PY_VER="38"
ARG PYTHON=python3

RUN yum update -y && yum install -y \
    python${PY_VER} \
    python${PY_VER}-pip \
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
ARG TF_WHEEL="tf_nightly-2.10.0.202218-cp38-cp38-linux_x86_64.whl"
ARG TF_ESTIMATOR_VER="2.10.0.dev2022042008"
ARG KERAS_NIGHTLY_VER="2.10.0.dev2022042007"

COPY ./whls/${TF_WHEEL} /tmp/pip3/

RUN python -m pip install --no-cache-dir \
    "tf-estimator-nightly==${TF_ESTIMATOR_VER}" \
    "keras-nightly==${KERAS_NIGHTLY_VER}" \
    /tmp/pip3/${TF_WHEEL}

RUN yum install -y https://extras.getpagespeed.com/release-el8-latest.rpm && \
    yum install -y gperftools && \
    yum erase -y getpagespeed-extras-release && \
    yum clean all

ENV LD_PRELOAD="/usr/lib64/libtcmalloc.so":${LD_PRELOAD}

# Please see: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-0778
RUN yum erase openssl -y && \
    yum clean all
