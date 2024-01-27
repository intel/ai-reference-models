# Running DLRM-v1 inference on Intel® Data Center GPU Flex Series using Intel® Extension for PyTorch*

## Overview

This document has instructions for running DLRM-v1 inference using Intel® Extension for PyTorch on Intel® Flex Series GPU.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html) 170  |
| Drivers | GPU-compatible drivers need to be installed: Download [Driver](https://dgpu-docs.intel.com/driver/installation.html) |
| Software | Docker* |

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Batch inference with FP16 precision on Flex series 170 |

> [!NOTE]
> At the moment sample does not support FP32 precision (`export PRECISION=fp16`).

## Datasets and Pre-trained Model

Refer to the [link](README.md#prepare-dataset) to download and prepare datasets and pre-trained models. Set `DATASET_DIR` and `CKPT_DIR` to point to the corresponding directories. 

## Run Using Docker

### Set up Docker Image

```bash
docker pull intel/recommendation:pytorch-flex-gpu-dlrm-v1-inference
```
### Run Docker Image
The DLRM-v1 inference container includes scripts, model and libraries needed to run FP16 inference. To run the `run_model.sh` quickstart script using this container, you will need to provide an output directory where log files will be written.

```bash
#Optional
export PRECISION=fp16
export BATCH_SIZE=<provide batch size otherwise (default: 32768)>
export NUM_ITERATIONS=<provide num_iterations otherwise (default: 20)>

#Required
export OUTPUT_DIR=<path to output directory>
export SCRIPT=run_model.sh
export MULTI_TILE=False
export PLATFORM=Flex
export DATASET_DIR=<path to processed dataset directory>
export CKPT_DIR=<path to pre-trained model>

IMAGE_NAME=intel/recommendation:pytorch-flex-gpu-dlrm-v1-inference
DOCKER_ARGS="--rm -it"

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env PRECISION=${PRECISION} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env PLATFORM=${PLATFORM} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env CKPT_DIR=${CKPT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${CKPT_DIR}:${CKPT_DIR} \
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
