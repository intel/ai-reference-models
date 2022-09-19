# Please see: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-0778
RUN yum erase openssl -y && \
    yum clean all
