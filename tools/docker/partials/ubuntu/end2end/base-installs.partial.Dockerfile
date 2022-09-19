ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
    ca-certificates \
    libgomp1 \
    numactl \
    patch \
    wget
