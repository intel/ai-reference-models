ARG MASKRCNN_DIR

RUN source ~/anaconda3/bin/activate pytorch && \
    cd ${MASKRCNN_DIR} && \
    cd maskrcnn-benchmark && \
    python setup.py install && \
    pip install onnx
