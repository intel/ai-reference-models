ENV PATH="~/conda/bin:${PATH}"
ENV KMP_AFFINITY="granularity=fine,compact,1,0"
ENV KMP_BLOCKTIME=1
ENV DNNL_PRIMITIVE_CACHE_CAPACITY=1024
ENV KMP_SETTINGS=1
ENV LD_PRELOAD="/workspace/lib/jemalloc/lib/libjemalloc.so:/root/conda/envs/pytorch/lib/libiomp5.so:$LD_PRELOAD"
ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
ENV BASH_ENV=/root/.bash_profile
WORKDIR /workspace/
RUN yum install -y numactl mesa-libGL && \
    yum clean all && \
    echo "export LD_PRELOAD=${LD_PRELOAD%%:}" >> /root/.bash_profile && \
    echo "source activate pytorch" >> /root/.bash_profile
