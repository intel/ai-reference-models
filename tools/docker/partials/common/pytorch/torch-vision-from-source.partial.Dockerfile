# Build Torch Vision
ARG TORCHVISION_VERSION=v0.8.0

RUN source ~/anaconda3/bin/activate pytorch && \
    git clone https://github.com/pytorch/vision && \
    cd vision && \
    git checkout ${TORCHVISION_VERSION} && \
    python setup.py install
