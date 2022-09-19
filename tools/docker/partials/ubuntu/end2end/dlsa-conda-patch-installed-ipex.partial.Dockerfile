RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y patch && \
    IPEX_PATH=$(find /opt/conda/lib -name 'intel_extension_for_pytorch') && \
    patch ${IPEX_PATH}/quantization/quantization_utils.py < ipex_int8_patch.diff
