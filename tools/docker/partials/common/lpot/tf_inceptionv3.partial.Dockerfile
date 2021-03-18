WORKDIR ${LPOT_SOURCE_DIR}/examples/tensorflow/image_recognition

RUN wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/inceptionv3_fp32_pretrained_model.pb
ENV PRETRAINED_MODEL=${PWD}/inceptionv3_fp32_pretrained_model.pb
