ARG BERT_DIR

RUN source activate pytorch && \
    cd ${BERT_DIR} && \
    pip install --upgrade pip && \
    pip install -r examples/requirements.txt && \
    pip install -e . && \
    pip install datasets accelerate tfrecord && \
    conda install openblas && \
    conda install faiss-cpu -c pytorch && \
    pip install transformers==4.9.0

RUN cd .. && \
    rm -rf ${BERT_DIR}
