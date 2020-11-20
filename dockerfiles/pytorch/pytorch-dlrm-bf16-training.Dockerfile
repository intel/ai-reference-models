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

RUN conda install -y -c intel/label/oneapibeta pytorch

RUN conda install -y -c intel/label/oneapibeta intel-extension-for-pytorch

RUN conda install -y -c intel/label/oneapibeta torch_ccl
ARG PYTHON_VERSION=3.7
ENV LD_LIBRARY_PATH="/opt/conda/lib/python${PYTHON_VERSION}/site-packages/ccl/lib/:${LD_LIBRARY_PATH}"

RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --fix-missing -y git && \
    python -m pip install onnx && \
    python -m pip install -e git+https://github.com/mlperf/logging@0.7.0-rc2#egg=logging && \
    conda install -y -c intel scikit-learn && \
    conda install -c conda-forge gperftools && \
    conda clean -a \
