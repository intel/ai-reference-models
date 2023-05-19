# PyTorch ResNet50_v1.5 Inference

## Description 

This document has instructions for running ResNet50 V1.5 Inference with INT8 precision using Intel® Extension for PyTorch on Intel®Max Series GPU. 

## Datasets

The [ImageNet](http://www.image-net.org/) validation dataset is used.

Download and extract the ImageNet2012 dataset from http://www.image-net.org/, then move validation images to labeled subfolders, using [the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

After running the data prep script, your folder structure should look something like this:

```
imagenet
└── val
    ├── ILSVRC2012_img_val.tar
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   ├── ILSVRC2012_val_00006697.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` directory should be set as the
`DATASET_DIR`
(for example: `export DATASET_DIR=/home/<user>/imagenet`).

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_block_format.sh` | Runs ResNet50 V1.5 INT8 inference (block format)|

Requirements:
* Host machine has Intel(R) Data Center Max Series GPU
* Follow instructions to install GPU-compatible driver [602](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-dc.html#step-1-add-package-repository)
* Docker

### Docker pull command:

```
docker pull intel/image-recognition:pytorch-max-gpu-resnet50v1-5-inference
```
The ResNet50 v1.5 inference container includes scripts,model and libraries need to run INT8 inference. To run the `inference_block_format.sh` quickstart script using this container, you'll need to set the environment variable and provide volume mounts for the ImageNet dataset if real dataset is required. Otherwise, the script uses dummy data. You will need to provide an output directory where log files will be written. 

```
export PRECISION=<export precision>
export DATASET_DIR=${PWD}/imagenet
export OUTPUT_DIR=${PWD}/logs
export BATCH_SIZE=<export batch size,default is 1024>
export NUM_ITERATIONS=<export number of iterations,default is 10>
IMAGE_NAME=intel/image-recognition:pytorch-max-gpu-resnet50v1-5-inference
DOCKER_ARGS="--rm -it"

SCRIPT=quickstart/inference_block_format.sh
Tile=2

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env Tile=${Tile} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
```

