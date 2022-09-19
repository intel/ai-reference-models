ARG PY_VER=38
RUN yum install -y gcc gcc-c++ && \
    yum install -y python${PY_VER}-devel && \
    yum clean all
