FROM centos-intel-base AS release
COPY --from=ipex-dev-base /root/conda /root/conda
COPY --from=ipex-dev-base /workspace/lib /workspace/lib

ENV LD_LIBRARY_PATH /lib64/:/usr/lib64/:/usr/local/lib64:/root/conda/envs/pytorch/lib:${LD_LIBRARY_PATH}
ENV PATH="~/conda/bin:${PATH}"
ENV DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
ENV BASH_ENV=/root/.bash_profile
WORKDIR /workspace/
RUN yum install -y numactl mesa-libGL && \
    yum clean all && \
    echo "source activate pytorch" >> /root/.bash_profile
