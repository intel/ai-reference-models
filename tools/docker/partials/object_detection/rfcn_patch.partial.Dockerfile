RUN cd ${TF_MODELS_DIR} && \
    git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b && \
    git apply --ignore-space-change --ignore-whitespace ${MODEL_WORKSPACE}/${PACKAGE_NAME}/models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch
