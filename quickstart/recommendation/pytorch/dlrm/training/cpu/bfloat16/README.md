<!--- 0. Title -->
# DLRM bf16 training

<!-- 10. Description -->
## Description

This document has instructions for running DLRM bf16 training using
Intel-optimized PyTorch.

<!--- 20. Datasets -->
## Datasets

Prepare your dataset according to the [instruction described here](/models/recommendation/pytorch/dlrm/training/bfloat16/README.md#4-prepare-dataset)

Set the `DATA_PATH` to point to "<dir/to/save/dlrm_data>" directory when running DLRM.

<!--- 30. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`train_single_node.sh`](train_single_node.sh) | Run 32K global BS with 4 ranks on 1 node (1 CPX6-4s Node). | 

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

<!--- 40. Bare Metal -->
## Bare Metal

To run on bare metal first, follow the [instruction described here](/models/recommendation/pytorch/dlrm/training/bfloat16/README.md#1-install-anaconda-30) until section 4.

After installing the prerequisites, download and untar the DLRM bf16 training model package, and run the quick start script:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/dlrm-bfloat16-training.tar.gz
tar -xvf dlrm-bfloat16-training.tar.gz
```
Set environment variables
for the path to your `DATA_PATH`then run a 
[quickstart script](#quick-start-scripts).

```
DATA_PATH=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

cd dlrm-bfloat16-training
bash quickstart/<script name>.sh
```

<!--- 50. Docker -->
## Docker

Use the base [PyTorch 1.9 container](https://hub.docker.com/r/intel/intel-optimized-pytorch/) 
`intel/intel-optimized-pytorch:1.9.0` to run DLRM bfloat16 training.
To run the quickstart scripts using the base PyTorch 1.9 container, install the model dependencies, and provide volume mounts for the DLRM bf16 training package and the dataset:

```
DATA_PATH=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATA_PATH=${DATA_PATH} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATA_PATH}:${DATA_PATH} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume <path to the model package directory>:/dlrm-bf16-training \
  --privileged --init -it \
  intel/intel-optimized-pytorch:1.9.0 /bin/bash 
```

Install the model dependencies inside the base PyTorch 1.9 container:
```
# MLPerf logging
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging

# MPICH
apt-get update && apt install mpich

# Model dependencies
pip install sklearn onnx tqdm

# torch-ccl:
git clone --single-branch --branch ccl_torch1.9 https://github.com/intel/torch-ccl.git && cd torch-ccl
git submodule sync 
git submodule update --init --recursive
python setup.py install

# tcmalloc
apt-get update && apt-get install wget
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz && \
tar -xzf gperftools-2.7.90.tar.gz && \
cd gperftools-2.7.90 && \
mkdir -p /workspace/lib/ && \
./configure --prefix=/workspace/lib/tcmalloc/ && \
make && \
make install
```

Run DLRM bfloat16 training script:
```
cd /dlrm-bf16-training
bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 70. License -->
## License

[LICENSE](/LICENSE)
