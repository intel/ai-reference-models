ARG IPEX_WHEEL_WW
ARG IPEX_WHEEL_YEAR
ARG CONDA_INSTALL_PATH=/opt/conda

RUN export PATH=${CONDA_INSTALL_PATH}/envs/dlsa/bin:${PATH} && \
    wget -r --no-parent -l1 --reject="index.html*" --cut-dirs=4 -nH http://mlpc.intel.com/downloads/cpu/${IPEX_WHEEL_YEAR}/${IPEX_WHEEL_YEAR}_ww${IPEX_WHEEL_WW}/ -A "torch-*.whl" -P /tmp && \
    wget -r --no-parent -l1 --reject="index.html*" --cut-dirs=4 -nH http://mlpc.intel.com/downloads/cpu/${IPEX_WHEEL_YEAR}/${IPEX_WHEEL_YEAR}_ww${IPEX_WHEEL_WW}/ -A "intel_extension_for_pytorch*.whl" -P /tmp && \
    pip install /tmp/torch-*.whl && \
    pip install /tmp/intel_extension_for_pytorch*.whl && \
    rm /tmp/torch-*.whl && \
    rm /tmp/intel_extension_for_pytorch*.whl
