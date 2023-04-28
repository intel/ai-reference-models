RUN yum --enablerepo=extras install -y epel-release && \
    yum install -y \
    ca-certificates \
    git \
    wget \
    make \
    cmake \
    gcc-c++ \
    gcc \
    autoconf \
    bzip2 \
    tar
