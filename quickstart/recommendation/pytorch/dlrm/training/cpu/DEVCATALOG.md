# PyTorch DLRM training

## Description 
This document has instructions for running DLRM Training using Intel Extension for PyTorch. 

## Pull Command

Docker image based on CentOS Stream8
```
docker pull intel/recommendation:centos-pytorch-cpu-dlrm-training
```

Docker image based on Ubuntu 22.04
```
docker pull intel/recommendation:ubuntu-pytorch-cpu-dlrm-training
```

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training.sh` | Run training for the specified precision (fp32, bf32 or bf16). |


## Datasets
### Criteo Terabyte Dataset

The [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) is used to run DLRM. To download the dataset, you will need to visit the Criteo website and accept their terms of use: [https://labs.criteo.com/2013/12/download-terabyte-click-logs/](https://labs.criteo.com/2013/12/download-terabyte-click-logs/). Copy the download URL into the command below as the `<download url>` and replace the `<dir/to/save/dlrm_data>` to any path where you want to download and save the dataset.
```
export DATASET_DIR=<dir/to/save/dlrm_data>

mkdir ${DATASET_DIR} && cd ${DATASET_DIR}
curl -O <download url>/day_{$(seq -s , 0 23)}.gz
gunzip day_*.gz
```
The raw data will be automatically preprocessed and saved as `day_*.npz` to the `DATASET_DIR` when DLRM is run for the first time. On subsequent runs, the scripts will automatically use the preprocessed data.

## Docker Run
(Optional) Export related proxy into docker environment.
```bash
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```
To run DLRM training, set environment variables to specify the dataset directory, precision,pre-trained model, and an output directory. 
```bash
export OS=<provide either centos or ubuntu>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRECISION=<specify the precision to run>
export SCRIPT=quickstart/training.sh
export NUM_BATCH=<10000 for test performance and 50000 for testing convergence trend>  

IMAGE_NAME=intel/recommendation:${OS}-pytorch-cpu-dlrm-training
DOCKER_ARGS="--privileged --init -it"
WORKDIR=/workspace/pytorch-dlrm-training

docker run --rm \
  --env NUM_BATCH=${NUM_BATCH} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_RUN_ENVS} \
  --shm-size 10G \
  -w ${WORKDIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
```
## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/recommendation)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/recommendation/pytorch/dlrm/training/cpu/DEVCATALOG.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pyt-cpu)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
