ENV USE_DAAL4PY_SKLEARN YES

RUN conda install -y -q \
        daal4py \
        scikit-learn \
        threadpoolctl && \
    conda clean -y --all
