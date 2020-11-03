ENV HOROVOD_WITHOUT_MXNET=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_TENSORFLOW=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    cmake \
    git

# Please see this issue: https://github.com/horovod/horovod/issues/2355
# TODO: When a release come out that includes this commit (possibly 0.20.4 or newwer)
# replace the line with: pip install horovod>0.20.3
RUN pip install git+https://github.com/horovod/horovod.git@bb4e4cf7
