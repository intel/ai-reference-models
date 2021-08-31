ARG HOROVOD_VERSION=87094a4

ENV HOROVOD_WITHOUT_MXNET=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_TENSORFLOW=1

# In case installing released versions of Horovod fail,and there is
# a working commit replace next set of RUN commands with something like:
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    cmake \
    git
RUN pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends --fix-missing \
#     cmake
#
# RUN pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
