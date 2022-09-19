ARG PY_VERSION
ARG CONDA_INSTALL_PATH=/opt/conda

SHELL ["/bin/bash", "-c"]

RUN export PATH=${CONDA_INSTALL_PATH}/bin:${PATH} && \
    conda create -yn dlsa python=${PY_VERSION} && \
    source activate dlsa && \
    conda install -y -c conda-forge gperftools && \
    conda install -y intel-openmp pip astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses && \
    conda clean -ya && \
    export PATH=/opt/conda/envs/dlsa/bin:${PATH} && \
    pip install transformers==4.16.2 datasets==1.18.3 numpy

ENV CONDA_PREFIX="/opt/conda/envs/dlsa"
ENV PATH="/opt/conda/bin:${CONDA_PREFIX}/bin:${PATH}"
