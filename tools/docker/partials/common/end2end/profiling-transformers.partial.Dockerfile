ENV WORKSPACE=/workspace

COPY . ${WORKSPACE}/profiling-transformers

WORKDIR ${WORKSPACE}/profiling-transformers

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
