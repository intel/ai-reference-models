ARG PY_VERSION=3

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
        build-essential \
        python${PY_VERSION}-dev
