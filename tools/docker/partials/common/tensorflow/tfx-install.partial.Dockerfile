RUN python -m pip install --upgrade pip && \
    python -m pip install \
        requests==2.24.0 && \
    python -m pip install --pre tfx && \
    python -m pip install \
        struct2tensor \
        tensorflow-ranking
