RUN yum update -y && \
    yum install -y openmpi openmpi-devel && \
    yum clean all

ENV PATH="/usr/lib64/openmpi/bin:${PATH}"

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/lib64/openmpi/bin/mpirun /usr/lib64/openmpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/lib64/openmpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/lib64/openmpi/bin/mpirun && \
    chmod a+x /usr/lib64/openmpi/bin/mpirun

# Configure OpenMPI to run good defaults:
RUN echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf
