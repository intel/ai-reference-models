RUN source activate pytorch && \
    cd ${MODEL_WORKSPACE}/${PACKAGE_NAME}/quickstart && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git checkout v3.0.2 && \
    git apply ../enable_ipex_for_squad.diff && \
    pip install -e ./ && \
    pip install -r examples/requirements.txt && \
    conda install intel-openmp &&  \
    mkdir -p /root/.local
