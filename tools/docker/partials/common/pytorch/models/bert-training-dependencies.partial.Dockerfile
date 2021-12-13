ARG TRANSFORMERS_COMMIT

RUN source activate pytorch && \
    pip install datasets==1.11.0 accelerate tfrecord && \
    conda install openblas && \
    conda install faiss-cpu -c pytorch && \
    conda install intel-openmp && \
    cd ${MODEL_WORKSPACE}/${PACKAGE_NAME}/quickstart && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git checkout ${TRANSFORMERS_COMMIT} && \
    git apply ../enable_optmization.diff && \
    python -m pip install --upgrade pip && \
    pip uninstall transformers -y && \
    pip install -e . && \
    mkdir -p /root/.local
