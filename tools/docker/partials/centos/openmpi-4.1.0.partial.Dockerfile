# Install OpenMPI
ARG OPENMPI_VERSION="openmpi-4.1.0"
ARG OPENMPI_DOWNLOAD_URL="https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.0.tar.gz"

RUN mkdir /tmp/openmpi && \
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
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
    chmod a+x /usr/local/bin/mpirun

# Configure OpenMPI to run good defaults:
RUN echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf
