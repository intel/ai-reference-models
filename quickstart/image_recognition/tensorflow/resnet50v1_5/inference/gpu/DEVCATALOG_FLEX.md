# Running ResNet50 v1.5 Inference with Int8 on Intel® Data Center GPU Flex Series using Intel® Extension for TensorFlow*

## Overview

This document has instructions for running ResNet50 v1.5 inference using Intel® Extension for TensorFlow* with Intel® Data Center GPU Flex Series.


## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Data Center GPU Flex Series  |
| Drivers | GPU-compatible drivers need to be installed: [Download Driver 555](https://dgpu-docs.intel.com/releases/stable_555_20230124.html#ubuntu-22-04)
| Software | Docker* Installed |

## Get Started

### Download Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5.

### Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `online_inference` | Runs online inference for int8 precision |
| `batch_inference` | Runs batch inference for int8 precision |
| `accuracy` | Measures the model accuracy for int8 precision |


## Run Using Docker

### Set up Docker Image

```
docker pull intel/image-recognition:tf-flex-gpu-resnet50v1-5-inference
```

### Run Docker Image
The ResNet50 v1-5 inference container includes scripts,model and libraries need to run int8 inference. To run one of the inference quickstart scripts using this container, you'll need to provide volume mounts for the ImageNet dataset for running `accuracy.sh` script. For `online_inference.sh` and `batch_inference.sh` dummy dataset will be used. You will need to provide an output directory where log files will be written. 
```
export PRECISION=int8
export OUTPUT_DIR=<path to output directory>
export DATASET_DIR=<path to the preprocessed imagenet dataset>
IMAGE_NAME=intel/image-recognition:tf-flex-gpu-resnet50v1-5-inference
export GPU_TYPE=flex_series

docker run \
  --device=/dev/dri \
  --ipc=host \
  --privileged \
  --env PRECISION=${PRECISION} \
  --env GPU_TYPE=${GPU_TYPE} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --rm -it \
  $IMAGE_NAME \
  /bin/bash quickstart/<script name>.sh
```

## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/dockerfiles/model_containers)

## Support
Support for Intel® Extension for TensorFlow* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
