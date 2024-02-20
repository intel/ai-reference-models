# Running UNet++ inference on Intel® Data Center GPU Flex Series using Intel® Extension for PyTorch*

## Overview

This document has instructions for running UNet++ inference using Intel® Extension for PyTorch on Intel® Flex Series GPU.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html) 170  |
| Drivers | GPU-compatible drivers need to be installed: Download [Driver](https://dgpu-docs.intel.com/driver/installation.html) |
| Software | Docker* |

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Inference with FP16 for specified batch size on Flex series 170 |

### Set up Docker Image

```bash
docker pull intel/image-segmentation:pytorch-flex-gpu-unetpp-inference
```
### Run Docker Image
The UNet++ inference container includes scripts, model and libraries needed to run fp16 inference. To run the quickstart script using this container, the script uses dummy data. You will need to provide an output directory where log files will be written. 

```bash
#Optional
export PRECISION=fp16
export BATCH_SIZE=<provide batch size, otherwise (default:8)>

#Required
export PLATFORM=Flex
export MULTI_TILE=False
export OUTPUT_DIR=<path to output logs directory>

IMAGE_NAME=intel/image-segmentation:pytorch-flex-gpu-unetpp-inference
SCRIPT=run_model.sh
DOCKER_ARGS="--rm -it"

docker run \
    --device=/dev/dri \
    --env BATCH_SIZE=${BATCH_SIZE} \
    --env MULTI_TILE=${MULTI_TILE} \
    --env PLATFORM=${PLATFORM} \
    --env OUTPUT_DIR=${OUTPUT_DIR} \
    --env PRECISION=${PRECISION} \
    --env http_proxy=${http_proxy} \
    --env https_proxy=${https_proxy} \
    --env no_proxy=${no_proxy} \
    --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
    ${DOCKER_ARGS} \
    ${IMAGE_NAME} \
    /bin/bash $SCRIPT
```
## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/flex-gpu)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
