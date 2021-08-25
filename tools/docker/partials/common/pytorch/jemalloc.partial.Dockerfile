# Build Jemalloc
ARG JEMALLOC_SHA=c8209150f9d219a137412b06431c9d52839c7272

RUN source activate pytorch && \
    git clone  https://github.com/jemalloc/jemalloc.git && \
    cd jemalloc && \
    git checkout ${JEMALLOC_SHA} && \
    ./autogen.sh && \
    mkdir /workspace/lib/ && \
    ./configure --prefix=/workspace/lib/jemalloc/ && \
    make && \
    make install
