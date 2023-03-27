ARG PY_VERSION=3.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    python${PY_VERSION}-dev
