# Install OpenSSH for MPI to communicate between containers
RUN yum update -y && yum install -y  \
    openssh-server \
    openssh-clients && \
    yum clean all
