RUN source activate pytorch && \
    cd ${MODEL_WORKSPACE}/${PACKAGE_NAME}/models/language_modeling/pytorch/rnnt/training/cpu && \
    pip install -r requirements.txt && \
    pip install unidecode inflect && \
    pip install --upgrade pip && \
    pip install librosa sox && \
    yum install -y libsndfile && \
    mkdir -p /root/.local && \
    git clone https://github.com/HawkAaron/warp-transducer && \
    cd warp-transducer && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cd ../pytorch_binding && \
    python setup.py install
