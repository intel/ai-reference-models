ARG PY_VERSION
ARG INTEL_PY_BUILD

RUN conda config --add channels intel && \
    conda install  -y -q intelpython${PY_VERSION}_core==${INTEL_PY_BUILD} python=${PY_VERSION}
