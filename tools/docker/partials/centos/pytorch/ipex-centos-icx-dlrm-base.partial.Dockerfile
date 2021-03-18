ARG IPEX_CONTAINER_TAG=ipex-icx-dlrm-base:centos7
FROM $IPEX_CONTAINER_TAG

SHELL ["/bin/bash", "-c"]

RUN echo "source activate pytorch" >> ~/.bash_profile
