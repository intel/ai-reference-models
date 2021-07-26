<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package and then run a [quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where checkpoint and log files will be written>

wget <package url>
tar -xvf <package name>
cd <package dir>

./quickstart/<script name>.sh
```

To run distributed training (one MPI process per socket) for better throughput,
set the MPI_NUM_PROCESSES var to the number of sockets to use. 
To run with multiple instances, these additional dependencies will need to be
installed in your environment:

* openmpi-bin
* openmpi-common
* openssh-client
* openssh-server
* libopenmpi-dev
* horovod==0.19.1

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where checkpoint and log files will be written>
MPI_NUM_PROCESSES=<number of sockets to use>

wget <package url>
tar -xvf <package name>
cd <package dir>

./quickstart/<script name>.sh
```
