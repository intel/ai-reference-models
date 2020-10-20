ARG TF_WAVENET_BRANCH

ARG FETCH_PR

ARG CODE_DIR=/tensorflow-wavenet

ENV TF_WAVENET_DIR=${CODE_DIR}

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y git && \
    git clone https://github.com/ibab/tensorflow-wavenet.git ${CODE_DIR} && \
    ( cd ${CODE_DIR} && \
    if [ ! -z "$FETCH_PR" ]; then git fetch origin ${FETCH_PR}; fi && \
    git checkout ${TF_WAVENET_BRANCH} )
