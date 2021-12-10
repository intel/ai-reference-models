RUN source activate pytorch && \
    cd ${MODEL_WORKSPACE}/${PACKAGE_NAME}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark && \
    python setup.py install && \
    pip install onnx
