ARG LPOT_SOURCE_DIR=/src/lpot
ARG LPOT_BRANCH=master

ENV LPOT_SOURCE_DIR=$LPOT_SOURCE_DIR

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y git && \
    git clone --single-branch --branch ${LPOT_BRANCH} https://github.com/intel/lpot.git ${LPOT_SOURCE_DIR}

WORKDIR ${LPOT_SOURCE_DIR}
