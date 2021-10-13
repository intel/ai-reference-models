ARG BERT_DIR
ARG BERT_PRE_TRAIN_DIR
ARG TRANSFORMERS_COMMIT

RUN source activate pytorch && \
    cd ${BERT_DIR} && \
    pip install --upgrade pip && \
    pip install -r examples/requirements.txt && \
    pip install -e . && \
    pip install datasets==1.11.0 accelerate tfrecord && \
    conda install openblas && \
    conda install faiss-cpu -c pytorch && \
    cd ${BERT_PRE_TRAIN_DIR} && \
    rm -rf transformers && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git reset --hard ${TRANSFORMERS_COMMIT} && \
    wget https://github.com/huggingface/transformers/pull/13714.diff && \
    git apply 13714.diff && \
    python -m pip install --upgrade pip && \
    pip uninstall transformers -y && \
    pip install -e . && \
    mkdir -p /root/.local

RUN cd .. && \
    rm -rf ${BERT_DIR}
