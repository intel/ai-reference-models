RUN yum update -y && \
    yum install -y gcc gcc-c++ cmake python3-tkinter libXext libSM && \
    yum clean all
