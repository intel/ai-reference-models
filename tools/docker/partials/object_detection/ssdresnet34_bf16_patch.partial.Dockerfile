RUN cd ${TF_MODELS_DIR} && \
    git apply --ignore-space-change --ignore-whitespace ${MODEL_WORKSPACE}/${PACKAGE_NAME}/models/object_detection/tensorflow/ssd-resnet34/training/bfloat16/tf-2.0.diff
