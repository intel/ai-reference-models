RUN source activate pytorch && \
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz && \
    tar -xzf gperftools-2.7.90.tar.gz && \
    cd gperftools-2.7.90 && \
    mkdir -p /workspace/lib/ && \
    ./configure --prefix=/workspace/lib/tcmalloc/ && \
    make && \
    make install
