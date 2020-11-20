
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        unzip \
        git \
        rsync \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up Bazel
ENV BAZEL_VERSION 3.0.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

WORKDIR ${MODEL_WORKSPACE}/${PACKAGE_NAME}

RUN git clone --single-branch --branch=r0.5 https://github.com/tensorflow/addons.git && \
    (cd addons && \
    git apply ${MODEL_WORKSPACE}/${PACKAGE_NAME}/models/language_translation/tensorflow/mlperf_gnmt/gnmt-v0.5.2.patch && \
    echo "y" | bash configure.sh  && \
    bazel build --enable_runfiles build_pip_pkg && \
    bazel-bin/build_pip_pkg artifacts && \
    pip install artifacts/tensorflow_addons-*.whl --no-deps) && \
    rm -rf ./addons
