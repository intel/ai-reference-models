ARG HOROVOD_VERSION=87094a4

ENV HOROVOD_WITHOUT_MXNET=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_TENSORFLOW=1

# In case installing released versions of Horovod fail,and there is
# a working commit replace next set of RUN commands with something like:
RUN yum update -y && yum install -y git make && \
    yum clean all
RUN python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}

# RUN python3 -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
