ARG PYTORCH_WHEEL
ARG IPEX_WHEEL
ADD models/binaries/$PYTORCH_WHEEL /tmp/
ADD models/binaries/$IPEX_WHEEL /tmp/

RUN pip install typing-extensions>=3.6.2.1 numpy>=1.16.6 urllib3==1.25.4 six && \
    ( for whl in ${PYTORCH_WHEEL} ${IPEX_WHEEL}; do \
    pip install /tmp/$whl && \
    rm /tmp/$whl; done )
