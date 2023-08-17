# Running EfficientNet Inference with Int8 on Intel® Data Center GPU Flex Series using Intel® Extension for TensorFlow*

## Overview

This document has instructions for running EfficientNet inference using Intel® Extension for TensorFlow* with Intel® Data Center GPU Flex Series.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Data Center GPU Flex Series  |
| Drivers | GPU-compatible drivers need to be installed: [Download Driver 647](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html)
| Software | Docker* Installed |

## Get Started

### Download Dataset
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
The folder that contains the `val` directory should be set as the`IMAGE_FILE`
(for example: `export IMAGE_FILE=/home/<user>/imagenet/ILSVRC2012_val_00006697.JPEG`).
### Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `batch_inference` | Runs EfficientNet B0,B3 batch inference for fp16 precision on Flex series 170 |

## Run Using Docker

### Set up Docker Image

```bash
docker pull intel/image-recognition:tf-flex-gpu-efficientnet-inference
```
### Run Docker Image

The EfficientNet inference container contains scripts,models and libraries needed to run fp16 inference. You will need to provide an output directory where log files will be written. Additionally, you will have to download and volume mount an image from the ImageNet dataset.

```bash
export IMAGE_NAME=intel/image-recognition:tf-flex-gpu-efficientnet-inference
export MODEL_NAME=<EfficientNetB0 or EfficientNetB3>
export BATCH_SIZE=<provide batch size. Default is 64>
export PRECISION=fp16
export OUTPUT_DIR=<path to output directory>
export IMAGE_FILE=<path to ImageNet Image file>

docker run \
  --device=/dev/dri \
  --ipc=host \
  --privileged \
  --env PRECISION=${PRECISION} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env MODEL_NAME=${MODEL_NAME} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env IMAGE_FILE=${IMAGE_FILE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume IMAGE_FILE:${IMAGE_FILE} \
  --rm -it \
  $IMAGE_NAME \
  /bin/bash quickstart/batch_inference.sh
  ```

  ## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/dockers/flex-gpu)

## Support
Support for Intel® Extension for TensorFlow* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
