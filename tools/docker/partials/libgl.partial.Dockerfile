RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
        libgl1-mesa-glx \
        libglib2.0-0
