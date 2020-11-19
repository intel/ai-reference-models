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
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG CONDA_INSTALL_PATH=/opt/conda

ARG MINICONDA_VERSION=4.7.12

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
        wget \
        ca-certificates && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p ${CONDA_INSTALL_PATH} && \
    rm miniconda.sh && \
    ln -s ${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". ${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH="${CONDA_INSTALL_PATH}/bin:${PATH}"

ARG PY_VERSION
ARG INTEL_PY_BUILD

RUN conda config --add channels intel && \
    conda install  -y -q intelpython${PY_VERSION}_core==${INTEL_PY_BUILD} python=${PY_VERSION}

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    python3-apt \
    software-properties-common

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
        gcc-8 \
        g++-8 && \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 && \
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

ENV MODIN_ENGINE ray

RUN python3 -m pip install \
    modin \
    ray

# Fix for this error: https://github.com/ray-project/ray/issues/6013
RUN sed -i.bak '/include_webui/ s/^#*/#/' ${CONDA_INSTALL_PATH}/lib/python3.7/site-packages/modin/engines/ray/utils.py

ENV USE_DAAL4PY_SKLEARN YES

RUN conda install -y -q \
        daal4py \
        scikit-learn \
        threadpoolctl && \
    conda clean -y --all

ARG PACKAGE_DIR=model_packages

ARG PACKAGE_NAME

ARG MODEL_WORKSPACE

# ${MODEL_WORKSPACE} and below needs to be owned by root:root rather than the current UID:GID
# this allows the default user (root) to work in k8s single-node, multi-node
RUN umask 002 && mkdir -p ${MODEL_WORKSPACE} && chgrp root ${MODEL_WORKSPACE} && chmod g+s+w,o+s+r ${MODEL_WORKSPACE}

ADD --chown=0:0 ${PACKAGE_DIR}/${PACKAGE_NAME}.tar.gz ${MODEL_WORKSPACE}

RUN chown -R root ${MODEL_WORKSPACE}/${PACKAGE_NAME} && chgrp -R root ${MODEL_WORKSPACE}/${PACKAGE_NAME} && chmod -R g+s+w ${MODEL_WORKSPACE}/${PACKAGE_NAME} && find ${MODEL_WORKSPACE}/${PACKAGE_NAME} -type d | xargs chmod o+r+x 

WORKDIR ${MODEL_WORKSPACE}/${PACKAGE_NAME}

# Test the final container
RUN python -c "import modin.pandas"
