RUN conda config --add channels intel \
    && conda install -y -q tensorflow \
    && conda clean --all
