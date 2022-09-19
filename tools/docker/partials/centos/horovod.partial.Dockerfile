ARG HOROVOD_VERSION=11c1389
ENV HOROVOD_WITHOUT_MXNET=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_TENSORFLOW=1 \
    HOROVOD_CPU_OPERATIONS=MPI \
    HOROVOD_WITH_MPI=1 \
    HOROVOD_WITHOUT_GLOO=1

# Install Horovod
RUN yum update -y && yum install -y git cmake gcc-c++ && \
    yum clean all

RUN python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
