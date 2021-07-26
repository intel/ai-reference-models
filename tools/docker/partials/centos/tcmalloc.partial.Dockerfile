RUN dnf -y install https://extras.getpagespeed.com/release-el8-latest.rpm && \
    dnf -y install gperftools && \
    yum clean all
