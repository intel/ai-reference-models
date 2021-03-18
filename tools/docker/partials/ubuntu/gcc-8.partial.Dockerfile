RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    python3-apt \
    software-properties-common

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
        gcc-8 \
        g++-8 && \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 && \
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
