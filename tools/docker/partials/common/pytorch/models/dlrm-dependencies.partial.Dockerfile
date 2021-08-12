ARG DLRM_DIR

RUN source ~/anaconda3/bin/activate pytorch && \
    cd ${DLRM_DIR} && \
    pip install -r dlrm/requirements.txt
