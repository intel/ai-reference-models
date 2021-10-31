ARG CONDA_INSTALL_PATH=/opt/conda

ARG MINICONDA_VERSION=4.7.12

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
        wget \
        ca-certificates && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p ${CONDA_INSTALL_PATH} && \
    rm miniconda.sh && \
    ln -s ${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". ${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH="${CONDA_INSTALL_PATH}/bin:${PATH}"
