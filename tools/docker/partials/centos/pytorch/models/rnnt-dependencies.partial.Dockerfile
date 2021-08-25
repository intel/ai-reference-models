ARG RNNT_DIR

RUN source activate pytorch && \
    conda install intel-openmp && \
    yum install -y libsndfile && \
    cd ${RNNT_DIR} && \
    cd training/rnn_speech_recognition/pytorch && \
    pip install -r requirements.txt && \
    pip install unidecode inflect

RUN source activate pytorch && \
    cd /workspace && \
    git clone https://github.com/HawkAaron/warp-transducer && \
    cd warp-transducer && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cd ../pytorch_binding && \
    python setup.py install
