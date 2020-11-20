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

ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION}

RUN apt-get -y update && \
    apt-get install -y \
        python3-dev \
        python3-pip \
        git \
        wget \
        unzip \
        numactl

RUN pip3 -q install pip --upgrade && \
    pip3 install -U virtualenv && \
    pip3 install jupyter

# Since the notebook does a git patch, we will need to set the user name and email. 
# This can be dummy since it is within the container
RUN git config --global user.email "you@example.com" && \
    git config --global user.name "Your Name"

RUN git clone https://github.com/IntelAI/models.git

WORKDIR models/docs/notebooks/perf_analysis

ARG TF_VERSION=2.3.0

# Create a virtual environment for stock TF
RUN virtualenv -p python3 ./venv-stock-tf 

# Install all the necessary libraries for stock TF
RUN . ./venv-stock-tf/bin/activate && \
    pip install \
        cxxfilt  \
        gitpython \
        tensorflow==${TF_VERSION} \
        ipykernel \
        matplotlib \
        pandas \
        psutil && \
    deactivate

# Create a Jupyter notebook kernel for stock TF with the name stock-tensorflow
RUN venv-stock-tf/bin/python -m ipykernel install --user --name=stock-tensorflow

# Create a virtual environment for Intel TF
RUN virtualenv -p python3 ./venv-intel-tf

# Install all the necessary libraries for Intel TF environment
RUN . ./venv-intel-tf/bin/activate && \
    pip install \
        cxxfilt  \
        gitpython \
        intel-tensorflow==${TF_VERSION} \
        ipykernel \
        matplotlib \
        pandas \
        psutil && \
    deactivate

# Create a Jupyter notebook kernel for Intel TF with the name intel-tensorflow
RUN venv-intel-tf/bin/python -m ipykernel install --user --name=intel-tensorflow

EXPOSE 8888

ENV LISTEN_IP=localhost

# Run Jupyter notebook
CMD jupyter notebook --port=8888 --no-browser --ip=${LISTEN_IP} --allow-root
