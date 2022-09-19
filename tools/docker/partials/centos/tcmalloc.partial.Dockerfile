RUN yum install -y https://extras.getpagespeed.com/release-el8-latest.rpm && \
    yum install -y gperftools && \
    yum erase -y getpagespeed-extras-release && \
    yum clean all

ENV LD_PRELOAD="/usr/lib64/libtcmalloc.so":${LD_PRELOAD}
