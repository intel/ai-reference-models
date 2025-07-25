# PyTorch RNN-T Training

## Description
This document has instructions for running RNNT training using Intel Extension for PyTorch.

## Pull Command

```
docker pull intel/language-modeling:centos-pytorch-cpu-rnnt-training
```

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `training.sh` | Runs training for the specified precision (fp32,bf32 or bf16). |

# Datasets 
Follow instructions to download and pre-process [instructions](https://github.com/IntelAI/models/blob/v2.9.0/quickstart/language_modeling/pytorch/rnnt/training/cpu/download_dataset.sh) and set `DATASET_DIR` variable to point to the dataset. 

# Pre-Trained Model
Follow [instructions](https://github.com/IntelAI/models/blob/v2.9.0/quickstart/language_modeling/pytorch/rnnt/training/cpu/download_dataset.sh) to download the RNNT pre-trained model.

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

To run RNNT training, set environment variables to specify the dataset directory, precision to run, model directory and an output directory. 

```bash
export OUTPUT_DIR=<specify path to output directory>
export PRECISION=<specify the precision>
export DATASET_DIR=<specify path to processed dataset>
export CHECKPOINT_DIR=<specify path to pre-trained model>

IMAGE_NAME=intel/language-modeling:centos-pytorch-cpu-rnnt-training
SCRIPT=quickstart/<specify script name>
WORKDIR=/workspace/pytorch-rnnt-training
DOCKER_ARGS="--privileged --init -it"

docker run --rm \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  ${DOCKER_RUN_ENVS} \
  --shm-size 8G \
  -w ${WORKDIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT $PRECISION
```
## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/language-modeling)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/quickstart/language_modeling/pytorch/rnnt/training/cpu/DEVCATALOG.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pyt-cpu)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
