RUN conda config --add channels intel \
    && conda install  -y -q xgboost \
    && conda clean --all
