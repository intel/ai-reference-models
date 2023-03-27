ARG TORCH_CCL_PATH=models/torch_ccl
ARG HOROVOD_WHEEL
ADD models/binaries/$HOROVOD_WHEEL /tmp/

ADD $TORCH_CCL_PATH /tmp/torch_ccl

RUN pip install /tmp/$HOROVOD_WHEEL && \
    cd /tmp/torch_ccl && \
    python setup.py install
