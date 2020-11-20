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

ARG INTEL_PYTHON_TAG=latest
FROM intelpython/intelpython3_core:$INTEL_PYTHON_TAG

RUN conda install -y -c intel/label/oneapibeta pytorch

RUN conda install -y -c intel/label/oneapibeta intel-extension-for-pytorch

RUN conda install -y -c intel/label/oneapibeta torch_ccl
ARG PYTHON_VERSION=3.7
ENV LD_LIBRARY_PATH="/opt/conda/lib/python${PYTHON_VERSION}/site-packages/ccl/lib/:${LD_LIBRARY_PATH}"

RUN python -m pip install onnx && \
    python -m pip install -e git+https://github.com/mlperf/logging@0.7.0-rc2#egg=logging && \
    conda install -y -c intel scikit-learn && \
    conda install -c conda-forge gperftools && \
    conda clean -a \
