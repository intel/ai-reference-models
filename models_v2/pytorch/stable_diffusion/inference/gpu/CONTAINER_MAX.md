# Running Stable Diffusion inference on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch*

## Overview

This document has instructions for running Stable Diffusion inference using Intel® Extension for PyTorch on Intel® Max Series GPU.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)  |
| Drivers | GPU-compatible drivers need to be installed: Download [Driver](https://dgpu-docs.intel.com/driver/installation.html) |
| Software | Docker* |

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Inference for specified precision FP16 with batch size 1 on Max Series |

## Run Using Docker

### Set up Docker Image

```
docker pull intel/generative-ai:pytorch-max-gpu-stable-diffusion-inference
```

### Run Docker Image
The stable diffusion inference container includes scripts,model and libraries need to run FP16 inference. To run the `run_model.sh` quickstart script using this container, you will need to provide an output directory where log files will be written. 

```bash
#Optiional
export PRECISION=fp16
export BATCH_SIZE=<provide batch size otherwise, batch size of 1 will be used>

#Required
export OUTPUT_DIR=<path to output directory>
export SCRIPT=run_model.sh
export MULTI_TILE=True
export PLATFORM=Max

IMAGE_NAME=intel/generative-ai:pytorch-max-gpu-stable-diffusion-inference
DOCKER_ARGS="--rm -it"

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env PLATFORM=${PLATFORM} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_ARGS} \
  ${IMAGE_NAME} \
  /bin/bash $SCRIPT
```
## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/max-gpu)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
