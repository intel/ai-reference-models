RUN conda install -y -c intel/label/oneapibeta torch_ccl
ARG PYTHON_VERSION=3.7
ENV LD_LIBRARY_PATH="/opt/conda/lib/python${PYTHON_VERSION}/site-packages/ccl/lib/:${LD_LIBRARY_PATH}"
