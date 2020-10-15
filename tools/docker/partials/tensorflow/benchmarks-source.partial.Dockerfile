ARG TF_BENCHMARKS_BRANCH

ARG TF_BENCHMARKS_DIR=/tensorflow/benchmarks

ENV TF_BENCHMARKS_DIR=${TF_BENCHMARKS_DIR}

RUN apt-get install --no-install-recommends --fix-missing -y git && \
    git clone --single-branch https://github.com/tensorflow/benchmarks.git ${TF_BENCHMARKS_DIR} && \
    ( cd ${TF_BENCHMARKS_DIR} && \
    git checkout ${TF_BENCHMARKS_BRANCH} )
