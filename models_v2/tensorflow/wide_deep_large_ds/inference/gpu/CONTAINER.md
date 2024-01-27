# Running Wide and Deep Inference with FP16 on Intel® Data Center GPU Flex Series using Intel® Extension for TensorFlow*

## Overview

This document has instructions for running Wide and Deep model inference using Intel® Extension for TensorFlow* with Intel® Data Center GPU Flex Series.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html) 170 or 140 |
| Drivers | GPU-compatible drivers need to be installed: [Download Driver](https://dgpu-docs.intel.com/driver/installation.html)
| Software | Docker* |

## Get Started

### Download Datasets

Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/large_kaggle_advertising_challenge/README.md) to download and preprocess the Large Kaggle Display Advertising Challenge Dataset.

Set the `DATASET_PATH` to point to the TF records directory.

### Pre-trained Model

Get the pre-trained model as follows. 
```bash
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_0/wide_deep_fp16_pretrained_model.pb
```
Set the `PB_FILE_PATH` to point to the downloaded model. 

### Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `run_model.sh` | Runs batch inference for fp16 precision on Flex 170 and Flex 140 |

## Run Using Docker

### Set up Docker Image

```bash
docker pull intel/recommendation:tf-flex-gpu-wide-and-deep-inference
```

### Run Docker Image

The Wide and Deep inference container includes scripts, model and libraries needed to run FP16 inference. To run the `run_model.sh` script using this container, you'll need to provide volume mounts for the dataset and pre-trained model. You will also need to provide an output directory where log files will be written. 

```bash
#Optional
BATCH_SIZE=<provide batch size. Default is 10000>

#Required
export DATASET_PATH=<path to processed dataset directory>
export PB_FILE_PATH=<path to pre-trained model>
export OUTPUT_DIR=<path to output logs directory>

GPU_TYPE=<provide either flex_170 or flex_140>
IMAGE_NAME=intel/recommendation:tf-flex-gpu-wide-and-deep-inference

docker run \
  --device=/dev/dri \
  --ipc=host \
  --privileged \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env GPU_TYPE=${GPU_TYPE} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_PATH=${DATASET_PATH} \
  --env PB_FILE_PATH=${PB_FILE_PATH} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_PATH}:${DATASET_PATH} \
  --volume ${PB_FILE_PATH}:${PB_FILE_PATH} \
  --rm -it \
  $IMAGE_NAME \
  /bin/bash run_model.sh
```
**Note:** Add `--cap-add=SYS_NICE` to the `docker run` command for executing the script on Flex series 140.
## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/flex-gpu)

## Support
Support for Intel® Extension for TensorFlow* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
