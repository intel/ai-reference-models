ARG DLRM_DIR

RUN source activate pytorch && \
    cd ${DLRM_DIR} && \
    pip install -r dlrm/requirements.txt
