ARG BERT_DIR

RUN source ~/anaconda3/bin/activate pytorch && \
    cd ${BERT_DIR} && \
    cd bert && \
    pip install -r examples/requirements.txt && \
    pip install . && \
    conda install -c conda-forge "llvm-openmp"
