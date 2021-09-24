ARG BASE_IMAGE=centos:8

FROM ${BASE_IMAGE} AS centos-intel-base
SHELL ["/bin/bash", "-c"]
