<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

wget <package url>
tar -xzf <package name>
cd <package dir>

./quickstart/fp32_training.sh (or ./quickstart/fp32_training_demo.sh)
```

For training in multi-instance mode (2 sockets in a single node for example) in evaluation mode,
where we are "saving checkpoints" and "doing the evaluation", the following prerequisites must be installed in your environment:
* gcc-8
* g++-8
* libopenmpi-dev
* openmpi
* openssh
* horovod

Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a
[quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

cd <package dir>
./quickstart/fp32_training_mpirun.sh
```

