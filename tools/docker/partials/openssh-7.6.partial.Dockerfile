RUN apt-get install --no-install-recommends --fix-missing -y \
    openssh-client \
    openssh-server \
    systemd && \
    systemctl enable ssh
