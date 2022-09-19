# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ARG PY_VER=38
ARG PYTHON=python3

RUN yum update -y && yum install -y \
    python${PY_VER} \
    python${PY_VER}-pip \
    which && \
    yum clean all

RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python

# Installs the latest version by default.
ARG TF_WHEEL=tf_nightly-2.10.0.202218-cp38-cp38-linux_x86_64.whl
ARG TF_ESTIMATOR_VER=2.10.0.dev2022042008
ARG KERAS_NIGHTLY_VER=2.10.0.dev2022042007

COPY ./whls/${TF_WHEEL} /tmp/pip3/

RUN python -m pip install --no-cache-dir \
    "tf-estimator-nightly==${TF_ESTIMATOR_VER}" \
    "keras-nightly==${KERAS_NIGHTLY_VER}" \
    /tmp/pip3/${TF_WHEEL}
