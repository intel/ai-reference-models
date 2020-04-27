# MiniGo

This document has instructions for how to run MiniGo for the
following modes/precisions:
* [FP32 training](#fp32-training-instructions)

Instructions and scripts for model training and inference
for other precisions are coming later.

## FP32 training Instructions
0. Minigo project will install a specific Tensorflow version, which may change the python environment. Thus, we recommend to use `conda` or `virtualenv` to create a separate environment for running Minigo project.

1. Clone the `minigo` repository with the specified SHA,
since we are using an specified version of the models repo for
MiniGo.
The minigo repo will be used for running training as well as 
download required files from Google Cloud Storage.

```
$ git clone --single-branch https://github.com/tensorflow/minigo.git --branch mlperf.0.6
$ cd minigo
$ git checkout 60ecb12f29582227a473fdc7cd09c2605f42bcd6
```

2. Obtain Minigo `checkpoint` and `target` data from Google Cloud Storage

2.1 (Optional) Install gsutil

If you have installed `gsutil` before, please skip this step. You may type command `gsutil --version` to 
check whether the `gsutil` has already been installed   

```bash
$ wget https://storage.googleapis.com/pub/gsutil.zip
$ tar xfz gsutil.tar.gz -C $HOME
$ export PATH=${PATH}:$HOME/gsutil
```

2.2 Download the `checkpoint` and `target` folders and copy them to the `minigo/mlperf` directory 
```bash
# under minigo directory
$ gsutil cp -r gs://minigo-pub/ml_perf/0.6/checkpoint ml_perf/
# organize target folders
$ cd ml_perf/target
$ mkdir 9
$ mv target* ./9
$ cd ../../
# organize checkpoint folders
$ gsutil cp -r gs://minigo-pub/ml_perf/0.6/target ml_perf/
$ cd ml_perf/checkpoint/
$ mv ./work_dir/work_dir/* ./work_dir/
$ rm -rf ./work_dir/work_dir/
$ mkdir 9
$ mv ./golden_chunks ./9
$ mv ./work_dir ./9
$ cd ../../../

```

The organized `checkpoint` folders are shown below. 

```
.
└── 9
    ├── golden_chunks
    │   ├── 000000-000000.tfrecord.zz
    │   ├── 000000-000001.tfrecord.zz
    │   ├── 000000-000002.tfrecord.zz
    │   ├── 000000-000003.tfrecord.zz
    │   ├── 000000-000004.tfrecord.zz
    │   ├── 000000-000005.tfrecord.zz
    │   ├── 000000-000006.tfrecord.zz
    │   ├── 000000-000007.tfrecord.zz
    │   ├── 000000-000008.tfrecord.zz
    │   └── 000000-000009.tfrecord.zz
    └── work_dir
        ├── checkpoint
        ├── model.ckpt-9383.data-00000-of-00001
        ├── model.ckpt-9383.index
        └── model.ckpt-9383.meta
```

The organized `target` folders are shown below. 
```
.
└── 9
    ├── target.data-00000-of-00001
    ├── target.index
    └── target.meta
```

3. Install the MPI kits

3.1 Install Intel MPI

If you have installed Intel MPI before, please skip this step.

Download and install the [Intel(R) MPI Library for Linux](https://software.intel.com/en-us/mpi-library/choose-download/linux). 
Once you have the l_mpi_2019.3.199.tgz downloaded, unzip it into /home/\<user\>/l_mpi directory.

If you want to make the silent installation. **Change the value of "ACCEPT_EULA" to "accept" 
in /home/\<user\>/l_mpi/l_mpi_2019.3.199/silent.cfg**, before start the silent installation.
Run `sh install.sh --silent silent.cfg` to complete the installation. The software is installed by default to "/opt/intel" location.

Otherwise, you can make the custom installation. Run `sh install.sh` and make the custom options.

```bash
$ tar -zxvf l_mpi_2019.3.199.tgz -C /home/<user>/l_mpi
$ cd /home/<user>/l_mpi/l_mpi_2019.3.199

# 1. Silent installation
# change the value of "ACCEPT_EULA" to "accept"
$ vim silent.cfg
$ sh install.sh --silent silent.cfg

# 2. Custom installation
$ sh install.sh
# Follow the instructions and complete the installation
```

3.2 Install mpi4py
```bash
# set the necessary environmental variables
$ source /<mpi-installed-path>/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
$ export PATH=/<mpi-installed-path>/intel/impi/2019.3.199/intel64/bin/:$PATH
$ export MPICC=/<mpi-installed-path>/intel/impi/2019.3.199/intel64/bin/mpiicc
$ export CC=icc
$ export CPPFLAGS=-DOMPI_WANT_MPI_INTERFACE_WARNING=0
# install the mpi4py package
$ pip install mpi4py
```


4. Install essential tools

Check you have installed all the tools before start training.

4.1 Install gcc

The project has been tested on gcc 8.2.0. We recommend to run the project with gcc > 7.2.0.
```
$ git clone -b releases/gcc-8.2.0 https://github.com/gcc-mirror/gcc.git
$ cd gcc
$ ./configure  --prefix=/path/to/gcc
$ make $$ make install
$ export PATH=/path/to/gcc/bin:$PATH
$ export LD_LIBRARY_PATH=/path/to/gcc/lib:$LD_LIBRARY_PATH

```

4.2 Install bazel 0.22.0

Currently, only bazel release 0.22.0 works Minigo training.

```
$ wget https://github.com/bazelbuild/bazel/releases/download/0.22.0/bazel-0.22.0-installer-linux-x86_64.sh
$ chmod 755 bazel-0.22.0-installer-linux-x86_64.sh 
$ ./bazel-0.22.0-installer-linux-x86_64.sh --prefix=/<user>/bazel
$ rm /root/.bazelrc
$ export PATH=/<user>/bazel/bin:$PATH 
```

4.3 Install zlib

```
$ wget https://www.zlib.net/zlib-1.2.11.tar.gz
$ tar -xzf https://www.zlib.net/zlib-1.2.11.tar.gz
$ cd zlib-1.2.11
$ ./configure --prefix=/path/to/zlib
$ make $$ make install
$ export C_INCLUDE_PATH=/path/to/zlib/include:$C_INCLUDE_PATH
$ export CPLUS_INCLUDE_PATH=/path/to/zlib/include:$CPLUS_INCLUDE_PATH
$ export LD_LIBRARY_PATH=/path/to/zlib/lib:$LD_LIBRARY_PATH
 
```

5. Clone the [intelai/models](https://github.com/intelai/models) repository. 
This repository has the launch script for running the model, which we will use in the next step.

```bash
$ git clone https://github.com/IntelAI/models.git
```

6. Environment variables setting
```
$ source /<mpi-installed-path>/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
$ export LD_LIBRARY_PATH=/<mpi-installed-path>/intel/compilers_and_libraries_2019.3.199/linux/mpi/intel64/libfabric/lib:$LD_LIBRARY_PATH
$ export FI_PROVIDER=tcp
 
```

7. Next, navigate to the `benchmarks` directory of the
[intelai/models](https://github.com/intelai/models) repo that was just
cloned in the previous step. MiniGo can be run for training.


7.1 Single-node training
You may run the scripts below to execute the single-node training. The flag `model-source-dir` is the repository cloned in step 1. The flag `steps` sets the num of iterations to run (1 training, 1 selfplay and 1 eval per step).
The default value for `step` is 30. The flag `quantization` sets to apply int8 quantization or not and its default value is True.

```
$ cd /home/<user>/models/benchmarks
$ python launch_benchmark.py \
    --model-source-dir /home/<user>/minigo \
    --model-name minigo \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    -- steps=30 quantization=True

```

7.2 Multi-node training (normal mode)

First, prepare `node_list.txt` to contain all node addresses. Each line for a single ip address. Then Copy the `node_list.txt` file to the `benchmarks` directory of the [intelai/models].

```
# file node_list.txt
192.168.30.81
192.168.30.82
192.168.30.83
192.168.30.84
192.168.30.85

# Caution: no blank space after ip address
# For the example node_list.txt 
# cat node_list.txt | wc -l  =>  5

```

Second, the host node where the program is launched must be able to SSH to all other hosts without any prompts. Verify that you can ssh to every other server without entering a password. To learn more about setting up passwordless authentication, please see [this page](http://www.linuxproblem.org/art_9.html).
Also ensure that port 52175 on each node is not occupied by any other process or blocked by firewall. 

Third, add the `multi_node` flag to specify the distributed training, and the `num-train-nodes` flag to specify the number of training nodes. The evaluation nodes and the selfplay nodes share the rest of nodes given in `node_list.txt`.
```
$ cd /home/<user>/models/benchmarks
$ python launch_benchmark.py \
    --model-source-dir /home/<user>/minigo \
    --model-name minigo \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    -- steps=30 quantization=True num-train-nodes=2 multi_node=True
    
```

Fourth, if you do run on a large scale system (typically more than 32 nodes), add the large_scale flag to enable large scale mode, and num-eval-nodes flag to specify number of evaluation nodes. The number of selfplay nodes are the rest of nodes given in node_list.txt. A typical ratio of train, eval and selfplay nodes are 8 : 4 : 48.   There is also an additional node for orchestrating, so a total 8+4+48+1=61 nodes needs to be used to achieve this ratio.
```
$ cd /home/<user>/models/benchmarks
$ python launch_benchmark.py \
    --model-source-dir /home/<user>/minigo \
    --model-name minigo \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    -- steps=30 quantization=True num-train-nodes=8 num-eval-nodes=4 multi_node=True large_scale=True
    
```

8.  Generally, the model convergences in ~20 steps (average of 10 runs).
The log files are saved in the value of `/home/<user>/minigo/results/$HOSTNAME`.
Below is a sample of outputs in `/home/<user>/minigo/results/$HOSTNAME`:

```
  data  eval.log  flags  models  mpi  rl_loop.log  selfplay.log  sgf  train.log  work_dir
```
