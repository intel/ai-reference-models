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

ARG PYT_BASE_IMAGE="intel/image-recognition"
ARG PYT_BASE_TAG="pytorch-max-gpu-resnet50v1-5-training"

FROM ${PYT_BASE_IMAGE}:${PYT_BASE_TAG}

# Install OpenSSH
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openssh-client \
        openssh-server && \
    rm  /etc/ssh/ssh_host_*_key \
        /etc/ssh/ssh_host_*_key.pub && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd

# https://github.com/openucx/ucx/issues/4742#issuecomment-584059909
ENV UCX_TLS=ud,sm,self,tcp
