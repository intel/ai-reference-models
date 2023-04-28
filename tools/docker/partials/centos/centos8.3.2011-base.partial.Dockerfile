ARG BASE_IMAGE=centos:centos8.3.2011

FROM ${BASE_IMAGE} AS centos-intel-base
SHELL ["/bin/bash", "-c"]
