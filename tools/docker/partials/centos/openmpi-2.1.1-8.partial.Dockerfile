RUN yum update -y && \
    yum install -y openmpi openmpi-devel openssh openssh-server && \
    yum clean all

ENV PATH="/usr/lib64/openmpi/bin:${PATH}"
