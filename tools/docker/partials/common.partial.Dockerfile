ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing python-tk libsm6 libxext6 -y && \
    pip install requests

