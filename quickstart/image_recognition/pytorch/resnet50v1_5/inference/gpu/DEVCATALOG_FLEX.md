# Running ResNet50 v1.5 Inference with Int8 on Intel® Data Center GPU Flex Series using Intel® Extension for PyTorch*


## Overview

This document has instructions for running ResNet50v1.5 inference using Intel® Extension for PyTorch on Intel® Flex Series GPU.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Data Center GPU Flex Series  |
| Drivers | GPU-compatible drivers need to be installed: [Download Driver 555](https://dgpu-docs.intel.com/releases/stable_555_20230124.html#ubuntu-22-04)
| Software | Docker* Installed |

## Get Started

## Download Datasets

The [ImageNet](http://www.image-net.org/) validation dataset is used.

Download and extract the ImageNet2012 dataset from http://www.image-net.org/, then move validation images to labeled subfolders, using [the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

A after running the data prep script, your folder structure should look something like this:

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
The folder that contains the `val` directory should be set as the`DATASET_DIR`
(for example: `export DATASET_DIR=/home/<user>/imagenet`).

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_block_format.sh` | Runs ResNet50 inference (block format) for the specified precision (int8) |

## Run Using Docker

### Set up Docker Image

```
docker pull intel/image-recognition:pytorch-flex-gpu-resnet50v1-5-inference
```
### Run Docker Image
The ResNet50 v1-5 inference container includes scripts,model and libraries need to run int8 inference. To run the `inference_block_format.sh` quickstart script using this container, you'll need to set the environment variable and provide volume mounts for the ImageNet dataset if real dataset is required. Otherwise, the script uses dummy data. You will need to provide an output directory where log files will be written. 

```
export PRECISION=int8
export OUTPUT_DIR=<path to output directory>
export DATASET_DIR=<path to the preprocessed imagenet dataset>
export SCRIPT=quickstart/inference_block_format.sh 
export BATCH_SIZE=<set batch size. Default is 1024>
export NUM_ITERATIONS=<set number of iterations. Default is 10>
DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/image-recognition:pytorch-flex-gpu-resnet50v1-5-inference

docker run \
  --device=/dev/dri \
  --ipc=host \
  --privileged \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_ITERATIONS=$NUM_ITERATIONS \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  ${DOCKER_ARGS} \
  ${IMAGE_NAME} \
  /bin/bash $SCRIPT
  ```

## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/dockerfiles/model_containers)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.