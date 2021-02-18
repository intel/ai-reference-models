ARG PY_VERSION=3

RUN yum update -y && \
    yum install -y \
       build-essential \
       python${PY_VERSION}-dev
