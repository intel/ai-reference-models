ARG HOROVOD_VERSION=0.22.1
ENV HOROVOD_WITHOUT_MXNET=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_TENSORFLOW=1 \
    HOROVOD_CPU_OPERATIONS=MPI \
    HOROVOD_WITH_MPI=1 \
    HOROVOD_WITHOUT_GLOO=1

# Install Horovod
RUN python3 -m pip install --no-cache-dir horovod==${HOROVOD_VERSION}

# In case installing released versions of Horovod fail,and there is
# a working commit replace next set of RUN commands with something like:
# ARG HOROVOD_VERSION=87094a4
# RUN yum update -y && yum install -y git make && \
#    yum clean all
# RUN python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
