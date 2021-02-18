ENV DEBIAN_FRONTEND=noninteractive

RUN yum update -y && \
    yum install -y \
        numactl \
        libXext \
        libSM \
        python3-tkinter && \
    pip install requests
