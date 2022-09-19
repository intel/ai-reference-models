ARG BASE_IMAGE=centos:8

FROM ${BASE_IMAGE} AS centos-intel-base
SHELL ["/bin/bash", "-c"]

# Fixe for “Error: Failed to download metadata for repo 'appstream': Cannot prepare internal mirrorlist: No URLs in mirrorlist"
RUN sed -i.bak '/^mirrorlist=/s/mirrorlist=/#mirrorlist=/g' /etc/yum.repos.d/CentOS-Linux-* && \
    sed -i.bak 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-* && \
    yum distro-sync -y && \
    yum --disablerepo '*' --enablerepo=extras swap centos-linux-repos centos-stream-repos -y && \
    yum distro-sync -y && \
    yum clean all
