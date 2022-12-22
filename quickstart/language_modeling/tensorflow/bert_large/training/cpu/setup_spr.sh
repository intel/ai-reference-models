#!/usr/bin/env bash
#
# Copyright (c) 2022 Intel Corporation
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

yum update -y && \
yum install -y gcc gcc-c++ cmake python3-tkinter libXext libSM && \
yum clean all

# Install OpenMPI
OPENMPI_VERSION="openmpi-4.1.0"
OPENMPI_DOWNLOAD_URL="https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.0.tar.gz"

mkdir /tmp/openmpi && \
cd /tmp/openmpi && \
curl -fSsL -O ${OPENMPI_DOWNLOAD_URL} && \
tar zxf ${OPENMPI_VERSION}.tar.gz && \
cd ${OPENMPI_VERSION} && \
./configure --enable-mpirun-prefix-by-default && \
make -j $(nproc) all && \
make install && \
ldconfig && \
cd / && \
rm -rf /tmp/openmpi

# Create a wrapper for OpenMPI to allow running as root by default
mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
echo '#!/bin/bash' > /usr/local/bin/mpirun && \
echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
chmod a+x /usr/local/bin/mpirun
