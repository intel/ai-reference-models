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
