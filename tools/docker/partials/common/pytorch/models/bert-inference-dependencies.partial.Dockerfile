RUN source activate pytorch && \
    cd ${MODEL_WORKSPACE}/${PACKAGE_NAME}/quickstart && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git checkout v4.18.0 && \
    git apply ../enable_ipex_for_squad.diff && \
    pip install -e ./ && \
    pip install -r examples/pytorch/language-modeling/requirements.txt && \
    pip install tensorboard && \
    conda install intel-openmp &&  \
    mkdir -p /root/.local
