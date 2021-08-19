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

FROM ubuntu:18.04

RUN apt-get update --yes \
    && apt-get install wget --yes && \
    rm -rf /var/lib/apt/lists/*

ENV USE_DAAL4PY_SKLEARN=YES
ENV USER modin
ENV UID 1000
ENV HOME /home/$USER

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --home $HOME \
    $USER

ENV CONDA_DIR ${HOME}/miniconda

SHELL ["/bin/bash", "--login", "-c"]

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh && \
    bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u && \
    "${CONDA_DIR}/bin/conda" init bash && \
    rm -f /tmp/miniconda3.sh && \
    echo ". '${CONDA_DIR}/etc/profile.d/conda.sh'" >> "${HOME}/.profile"

RUN conda update -n base -c defaults conda -y && \
    conda create --name intel_sklearn --yes -c intel python=3.7 scikit-learn

COPY rcv1_svm.py "${HOME}/rcv1_svm.py"
COPY rcv1_loader.py "${HOME}/rcv1_loader.py"

RUN conda activate intel_sklearn && python ${HOME}/rcv1_loader.py

ENTRYPOINT ["/bin/bash", "--login", "-c", "conda run \"$@\"", "/bin/bash", "-n", "intel_sklearn", "/usr/bin/env", "--"]

CMD ["python", "${HOME}/rcv1_svm.py"]
