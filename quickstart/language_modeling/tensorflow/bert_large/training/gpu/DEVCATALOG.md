# TensorFlow BERT Large Training 

## Description
This document has instructions for running BERT Large training with BF16 precision using Intel(R) Extension for TensorFlow on Intel® Data Center GPU Max Series.

## Datasets

### Pre-trained models

Download and extract the bert large uncased (whole word masking) pre-trained model checkpoints from the [google bert repo](https://github.com/google-research/bert#pre-trained-models). The extracted directory should be set to the `BERT_LARGE_DIR` environment variable when running the quickstart scripts. A dummy dataset will be auto generated and  used for training scripts.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `bfloat16_training.sh` | Runs BERT Large BF16 training on two tiles|
| `bfloat16_training_hvd.sh` | Runs BF16 Distributed Training using Intel(R) Optimized Horovod on two tiles | 

## Docker
Requirements:
* Host machine has Intel(R) Data Center Max Series GPU
* Follow instructions to install GPU-compatible driver [602](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-dc.html#step-1-add-package-repository)
* Docker

## Docker pull Command
```
docker pull intel/language-modeling:tf-max-gpu-bert-large-training
```
The BERT Large training container includes scripts,models,libraries needed to run BF16 training. 

```
export BERT_LARGE_DIR=<path to pretrained model checkpoints>
export OUTPUT_DIR=<path to output log files>
IMAGE_NAME=intel/language-modeling:tf-max-gpu-bert-large-training
DOCKER_ARGS="--rm -it"
export SCRIPT=bfloat16_training.sh
export PRECISION=bfloat16

if [[ ${SCRIPT} == bfloat16_training.sh ]]; then
   export Tile="${Tile:-2}"
else
   export Tile=1
fi

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env BERT_LARGE_DIR=${BERT_LARGE_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env Tile=${Tile} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --env PRECISION=${PRECISION} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${BERT_LARGE_DIR}:${BERT_LARGE_DIR} \
  --volume /dev/dri:/dev/dri \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash quickstart/$SCRIPT
  ```