# Running ResNet50_v1.5 Training on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch*

## Description 
This document has instructions for running ResNet50 v1.5 training on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)  |
| Drivers | GPU-compatible drivers need to be installed: Download [Driver](https://dgpu-docs.intel.com/driver/installation.html) |
| Software | Docker* |

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Runs ResNet50 V1.5 BF16,FP32 and TF32 training on single or multiple GPU devices|

## Datasets
Download and extract the ImageNet2012 training and validation dataset from [http://www.image-net.org/ (http://www.image-net.org/),then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

After running the data prep script and extracting the images, your folder structure
should look something like this:
```bash
imagenet
├── train
│   ├── n02085620
│   │   ├── n02085620_10074.JPEG
│   │   ├── n02085620_10131.JPEG
│   │   ├── n02085620_10621.JPEG
│   │   └── ...
│   └── ...
└── val
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` and `train` directories should be set as the `DATASET_DIR` environment variable before running the quickstart script.

### Docker pull command:
```bash
docker pull intel/image-recognition:pytorch-max-gpu-resnet50v1-5-training
```
The ResNet50 v1.5 training container includes scripts, model and libraries needed to run BF16,FP32 and TF32 training. To run the `run_model.sh` quickstart script using this container, you'll need to provide volume mounts for the ImageNet dataset. You will need to provide an output directory where log files will be written. 

```bash
#Optional
export PRECISION=<provide either BF16,FP32 or TF32, otherwise (default: BF16)>
export BATCH_SIZE=<provide batch size, otherwise (default: 256)>
export NUM_ITERATIONS=<provide number of iterations,otherwise (default: 20)>

#Required
export DATASET_DIR=<path to ImageNet dataset>
export OUTPUT_DIR=<path to output logs directory>
export PLATFORM=Max
export MULTI_TILE=<provide True for multi-tile GPU such as Max 1550, and False for single-tile GPU such as Max 1100>
export NUM_DEVICES=<provide the number of GPU devices used for training. It must be equal to or smaller than the number of GPU devices attached to each node. For GPU with 2 tiles, such as Max 1550 GPU, the number of GPU devices in each node is 2 times the number of GPUs, so it can be set as <=16 for a node with 8 Max 1550 GPUs. While for GPU with single tile, such as Max 1100 GPU, the number of GPU devices available in each node is the same as number of GPUs, so it can be set as <=8 for a node with 8 Max 1100 GPUs.>

DOCKER_ARGS="--rm --init -it"
IMAGE_NAME=intel/image-recognition:pytorch-max-gpu-resnet50v1-5-training
SCRIPT=run_model.sh

docker run \
  --device=/dev/dri \
  --ipc=host \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env PLATFORM=${PLATFORM} \
  --env NUM_DEVICES=${NUM_DEVICES} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env PRECISION=${PRECISION} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume /dev/dri:/dev/dri \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
  ```
## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/max-gpu)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html). Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
