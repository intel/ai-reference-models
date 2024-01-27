# Running ResNet50 v1.5 Inference with Int8 on Intel® Data Center GPU Flex Series using Intel® Extension for TensorFlow*

## Overview

This document has instructions for running ResNet50 v1.5 inference using Intel® Extension for TensorFlow* with Intel® Data Center GPU Flex Series.


## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html) 170 or 140 |
| Drivers | GPU-compatible drivers need to be installed: [Download Driver](https://dgpu-docs.intel.com/driver/installation.html)
| Software | Docker* |

## Get Started

### Download Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5.

### Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `run_model.sh` | Runs inference for int8 precision |

## Run Using Docker

### Set up Docker Image

```
docker pull intel/image-recognition:tf-flex-gpu-resnet50v1-5-inference
```

### Run Docker Image
The ResNet50 v1-5 inference container includes scripts, model and libraries needed to run int8 inference. To run the `run_model.sh` script using this container, you'll need to provide volume mounts for the ImageNet dataset for measuring accuracy test mode. For inference test mode, dummy dataset will be used.  You will need to provide an output directory where log files will be written. 

**Note:** The default batch size for Flex series 140 is 256 for batch inference and 1024 for Flex series 170. Additionally, add `--cap-add=SYS_NICE` to the `docker run` command for executing the script on Flex series 140.
```bash
#Optional 
BATCH_SIZE=<provide a batch size. Otherwise default batch sizes will be used>

#Required
export PRECISION=int8
export OUTPUT_DIR=<path to output directory>
export DATASET_DIR=<path to the preprocessed imagenet dataset>
export PB_FILE_PATH=/workspace/tf-flex-series-resnet50v1-5-inference/models/pretrained_models/resnet50v1_5-frozen_graph-int8-gpu.pb
IMAGE_NAME=intel/image-recognition:tf-flex-gpu-resnet50v1-5-inference
TEST_MODE=<provide either inference or accuracy>
FLEX_GPU_TYPE=<provide either flex_170 or flex_140 for inference mode>
SCRIPT=run_model.sh

docker run \
  --device=/dev/dri \
  --ipc=host \
  --privileged \
  --env PRECISION=${PRECISION} \
  --env FLEX_GPU_TYPE=${FLEX_GPU_TYPE} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env TEST_MODE=${TEST_MODE} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env PB_FILE_PATH=${PB_FILE_PATH} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --rm -it \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
```

## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/flex-gpu)

## Support
Support for Intel® Extension for TensorFlow* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
