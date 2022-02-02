RUN yum -y install https://extras.getpagespeed.com/release-el8-latest.rpm && \
    yum -y install gperftools && \
    yum clean all
