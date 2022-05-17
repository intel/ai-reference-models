ARG PYTORCH_IMAGE="intel/intel-optimized-pytorch"
ARG PYTORCH_TAG

FROM ${PYTORCH_IMAGE}:${PYTORCH_TAG} AS intel-optimized-pytorch
