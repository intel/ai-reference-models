# Steps to generate a container with Intel® Optimization for TensorFlow

This guide will help you generate a container with Intel's cpx release candidate.

## Steps:

1. Clone intel-tensorflow bf16/base branch:

    ```
    $ git clone https://github.com/Intel-tensorflow/tensorflow.git --branch=bf16/base --single-branch
    $ cd tensorflow
    $ git checkout 34305e8bf7ad4e7ffb9339b434385e3b28896924
    # Run "git log" and check for the right git hash
    ```

2.  Go to the directory that has Intel mkl docker files:

    ```
    $ cd tensorflow/tools/ci_build/linux/mkl/
    ```

3.  Run build-dev-container.sh by passing the following env parameters:

    ```
    $ env  ROOT_CONTAINER=tensorflow/tensorflow \
	ROOT_CONTAINER_TAG=devel \
	TF_DOCKER_BUILD_DEVEL_BRANCH=34305e8bf7ad4e7ffb9339b434385e3b28896924 \
	TF_REPO=https://github.com/Intel-tensorflow/tensorflow \
	TF_DOCKER_BUILD_VERSION=tensorflow-2.2-bf16-nightly \
	BUILD_SKX_CONTAINERS=yes \
	BUILD_TF_V2_CONTAINERS=yes \
	BUILD_TF_BFLOAT16_CONTAINERS=yes \
	BAZEL_VERSION= \
	ENABLE_DNNL1=yes \
	ENABLE_SECURE_BUILD=yes \
        ./build-dev-container.sh > ./container_build.log
    ```

4.  Open a second terminal session at the same location and run `tail -f container_build.log` to monitor container build progress
    or wait until the build finishes and then open the log file <container_build.log> ...

    ```
    INFO: Build completed successfully, 18811 total actions.
    ```

    Below output indicates that the container has intel-optimized tensorflow:

    ```
    PASS: MKL enabled test in <intermediate container name>
    ```

5.  Check if the image was built successfully and tag it:

    ```
    $ docker images
    intel-mkl/tensorflow:tensorflow-2.2-bf16-nightly-avx512-devel-mkl

    $ docker tag intel-mkl/tensorflow:tensorflow-2.2-bf16-nightly-avx512-devel-mkl intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly
    ```

6.  Run the image in privileged mode and install OpenMPI, OpenSSH and Horovod:
    Example of docker run command:

    ```
    $ docker run --init --privileged -it --env <Proxy setup or anything else> -v <mount dir>  --name container_name <imageid> /bin/bash
    ```

    Install Open MPI

    ```
    $ apt-get clean && apt-get update -y
    $ apt-get install -y --no-install-recommends --fix-missing openmpi-bin openmpi-common libopenmpi-dev

    # Check OpenMPI installation:
    $ mpirun --version
    # You should see the following message:
    mpirun (Open MPI) 2.1.1
    ```
    Install OpenSSH for MPI to communicate between containers
    ```

    $ apt-get install -y --no-install-recommends --fix-missing openssh-client openssh-server libnuma-dev
    $ mkdir -p /var/run/sshd
    # Allow OpenSSH to talk to containers without asking for confirmation
    $ cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new
    $ echo " StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new
    $ mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
    ```
    Install Horovod

    ```
    $ HOROVOD_WITH_TENSORFLOW=1
    $ python3 -m pip install --no-cache-dir horovod==0.19.1
    ```

    If Horovod installation was successful you will see the following message:

    ```
    Successfully installed cffi-1.14.0 cloudpickle-1.4.1 horovod-0.19.1 psutil-5.7.0 pycparser-2.20 pyyaml-5.3.1
    ```

    Check Horovod installation:

    ```
    $ python -c "import tensorflow as tf; import horovod.tensorflow as hvd;"
    ```
    You should not see an error.

7.  Save this image:

    ```
    $ exit
    $ docker ps -a
    $ docker commit [container ID] intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly
    ```

Substitute this docker image when using the parameter `--docker-image` in running benchmarks [`launch_benchmark.py`](/benchmarks/launch_benchmark.py).
