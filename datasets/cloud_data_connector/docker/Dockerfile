#
# -*- coding: utf-8 -*-
#
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
#

# Reference image
FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

# Runtime variables
ARG DEBIAN_FRONTEND=noninteractive

ENV MINIFORGE_VERSION="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
ENV PATH="${PATH}:/opt/conda/bin"

# Adding metadata to image
# LABEL label_example="Hi, this is a label example, is used for adding metadata"

# Create home workdir
WORKDIR /_work

# Install basic apps
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    wget \
    libicu-dev \
    gawk

RUN apt-get install -y \
    libsndfile1-dev \
    time

RUN apt-get clean
# RUN apt -y install ncurses-term
RUN apt-get install ncurses-term -y

RUN curl --output conda_installer.sh --location $MINIFORGE_VERSION
RUN bash conda_installer.sh -b -p /opt/conda
RUN rm conda_installer.sh

# Conda tools
RUN conda init bash
RUN conda config --set auto_update_conda False
RUN conda install mamba conda-build conda-verify -n base -c conda-forge --yes --quiet
RUN conda update --update-all -n base --yes --quiet
RUN conda clean --all --yes
RUN conda build purge-all
