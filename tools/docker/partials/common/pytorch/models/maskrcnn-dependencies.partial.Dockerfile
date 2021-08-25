ARG MASKRCNN_DIR

RUN source activate pytorch && \
    cd ${MASKRCNN_DIR} && \
    cd maskrcnn-benchmark && \
    python setup.py install && \
    pip install onnx
