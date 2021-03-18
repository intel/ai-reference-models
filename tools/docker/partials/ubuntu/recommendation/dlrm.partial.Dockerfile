ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get install --no-install-recommends --fix-missing -y git && \
    python -m pip install onnx && \
    python -m pip install -e git+https://github.com/mlperf/logging@0.7.0-rc2#egg=logging && \
    conda install -y -c intel scikit-learn && \
    conda install -c conda-forge gperftools && \
    conda clean -a
