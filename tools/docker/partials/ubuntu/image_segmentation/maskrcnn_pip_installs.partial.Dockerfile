ARG MASK_RCNN_SOURCE_DIR=/workspace/Mask_RCNN

ENV MODEL_SRC_DIR=${MASK_RCNN_SOURCE_DIR}

RUN pip install \
        IPython[all] \
        'Pillow>=8.1.2' \
        cython \
        h5py \
        imgaug \
        keras==2.0.8 \
        matplotlib \
        numpy==1.16.3 \
        opencv-python \
        pycocotools \
        scikit-image \
        scipy==1.2.0 && \
    apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
        git \
        wget

RUN git clone https://github.com/matterport/Mask_RCNN.git ${MODEL_SRC_DIR} && \
    ( cd ${MODEL_SRC_DIR} && \
    wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 )
