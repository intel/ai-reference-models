ARG BASE_IMAGE=centos:8

FROM ${BASE_IMAGE} AS centos-intel-base
SHELL ["/bin/bash", "-c"]

# This is to fix the yum command within CentOS 8
RUN sed -i '/^mirrorlist=/s/mirrorlist=/#mirrorlist=/g' /etc/yum.repos.d/CentOS-Linux-* && \
    sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-*
