RUN source ~/anaconda3/bin/activate pytorch && \
    pip install matplotlib Pillow pycocotools && \
    pip install yacs opencv-python cityscapesscripts transformers && \
    conda install -y libopenblas psutil && \
    cd /workspace/installs && \
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz && \
    tar -xzf gperftools-2.7.90.tar.gz && \
    cd gperftools-2.7.90 && \
    ./configure --prefix=$HOME/.local && \
    make && \
    make install && \
    rm -rf /workspace/installs/
