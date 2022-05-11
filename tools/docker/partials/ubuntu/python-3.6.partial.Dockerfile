ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get install -y python3.6 && \
    ln -s /usr/bin/python3.6 /usr/bin/python
