# TensorFlow ResNet50_v1.5 Training

## Description

This document has instructions for running ResNet50 v1.5 training with BFloat16 precision using Intel® Extension for TensorFlow on Intel® Data Center GPU Max Series.

## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md). After running the conversion script you should have a directory with the ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5.

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `bfloat16_training_full.sh` | bfloat16 precision script for ResNet50 v1.5 training on two tiles |
| `bfloat16_training_hvd.sh`| bfloat16 precision script for ResNet50 v1.5 with Intel® Optimization for Horovod* support on two tiles |

Requirements:
* Host machine has Intel(R) Data Center Max Series GPU
* Follow instructions to install GPU-compatible driver [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html#ubuntu-22-04)
* Docker

### Docker pull command:

```
docker pull intel/image-recognition:tf-max-gpu-resnet50v1-5-training
```
The ResNet50 v1.5 training container includes scripts,model and libraries need to run BFloat16 Training. To run the quickstart scripts using this container, you'll need to provide volume mounts for the ImageNet dataset.

```
export PRECISION=bfloat16
export OUTPUT_DIR=<path to log file directory>
export DATASET_DIR=<path to ImageNet dataset>

IMAGE_NAME=intel/image-recognition:tf-max-gpu-resnet50v1-5-training
DOCKER_ARGS="--rm -it"
export SCRIPT=bfloat16_training_hvd.sh

if [[ ${SCRIPT} == bfloat16_training_full.sh ]]; then
   export Tile=2
else
   export Tile=1
fi

docker run --rm \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env Tile=${Tile} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume /dev/dri:/dev/dri \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash quickstart/$SCRIPT
  ```
