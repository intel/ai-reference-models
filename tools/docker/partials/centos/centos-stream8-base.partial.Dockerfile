ARG BASE_IMAGE=quay.io/centos/centos:stream8

FROM ${BASE_IMAGE} AS centos-intel-base
SHELL ["/bin/bash", "-c"]
