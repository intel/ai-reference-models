# Copyright (c) 2020 Intel Corporation
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

ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION}

ARG CONDA_INSTALL_PATH=/databricks/conda

ARG MINICONDA_VERSION=4.8.3

# See https://github.com/databricks/containers/blob/master/ubuntu/minimal/Dockerfile
RUN apt-get update && \
    apt-get install --yes --no-install-recommends --fix-missing \
        bash \
        ca-certificates \
        coreutils \
        iproute2 \
        libc6 \
        openjdk-8-jdk-headless \
        procps \
        sudo \
        wget && \
    /var/lib/dpkg/info/ca-certificates-java.postinst configure && \
    rm -rf /var/lib/apt/lists/*

ENV PATH ${CONDA_INSTALL_PATH}/bin:$PATH

RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-py37_${MINICONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /databricks/conda && \
    rm miniconda.sh && \
    # Source conda.sh for all login and interactive shells.
    ln -s ${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /etc/profile.d/conda.sh" >> ~/.bashrc && \
    # Set always_yes for non-interactive shells.
    conda config --system --set always_yes True && \
    conda clean --all

COPY intel.yml /tmp/env.yml

RUN conda env create --file /tmp/env.yml && \
    rm -f /tmp/env.yml && \
    rm -rf $HOME/.cache/pip/*

RUN conda install -n intel -c intel scipy=1.5.2 --force-reinstall

ENV USE_DAAL4PY_SKLEARN=YES

# Set an environment variable used by Databricks to decide which conda environment to activate by default.
ENV DEFAULT_DATABRICKS_ROOT_CONDA_ENV=intel
