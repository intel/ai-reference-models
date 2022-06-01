# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# TensorFlow Dockerfile Development Container
#
# You can use this image to quickly develop changes to the Dockerfile assembler
# or set of TF Docker partials. See README.md for usage instructions.

FROM ubuntu:20.04

LABEL maintainer="Austin Anderson <angerson@google.com>"

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}

RUN if [ "$HTTPS_PROXY" != "" ]]; then \
  touch /etc/apt/apt.conf.d/proxy.conf && \
  echo 'Acquire::https::Proxy "'$HTTPS_PROXY'";' >> /etc/apt/apt.conf.d/proxy.conf; \
fi
RUN if [ "$HTTP_PROXY" != "" ]]; then \
  echo 'Acquire::http::Proxy "'$HTTP_PROXY'";' >> /etc/apt/apt.conf.d/proxy.conf; \
fi

RUN apt-get update && \
    apt-get install -y \
    bash \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    python3 \
    python3-pip \
    python-dev

RUN curl -sSL https://get.docker.com/ | sh
RUN pip3 install --upgrade pip==20.3.4 && \
    pip3 install \
         absl-py \
         cerberus \
         'cryptography<=3.2.1' \
         'docker==4.2.2' \
         GitPython \
         ndg-httpsclient \
         pyasn1 \
         pyopenssl \
         pyyaml \
         setuptools \
         urllib3

RUN curl -L -o kustomize.tar.gz https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv3.8.7/kustomize_v3.8.7_linux_amd64.tar.gz && \
    tar xzf kustomize.tar.gz && \
    chmod +x kustomize && \
    mv kustomize /usr/local/bin

WORKDIR /tf
VOLUME ["/tf"]

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
