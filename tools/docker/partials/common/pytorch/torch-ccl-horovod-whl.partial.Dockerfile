ARG TORCH_CCL_WHEEL
ARG HOROVOD_WHEEL
ADD models/binaries/$TORCH_CCL_WHEEL /tmp/
ADD models/binaries/$HOROVOD_WHEEL /tmp/

RUN pip install /tmp/$TORCH_CCL_WHEEL && \
    pip install /tmp/$HOROVOD_WHEEL && \
    rm /tmp/$TORCH_CCL_WHEEL && \
    rm /tmp/$HOROVOD_WHEEL
