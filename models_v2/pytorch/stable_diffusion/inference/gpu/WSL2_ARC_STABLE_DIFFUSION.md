# Running Stable Diffusion inference on Intel® Arc Series GPU using Intel® Extension for PyTorch*

## Overview

This document has instructions for running Stable Diffusion inference using Intel® Extension for PyTorch on Intel® Arc Series GPU on Windows 11 with WSL2.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Arc Series A770 GPU |
| Operating System | Windows 11 with [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) | 
| Drivers | [GPU drivers](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html) |
| Software | [Docker Desktop](https://docs.docker.com/desktop/wsl) |

Follow addtional pre-requisite [steps](WSL2_ARC_BASE.md) when accessing WSL2 remotely. 

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Inference with batch size 1 on Arc Series A770 |

> [!NOTE]
> At the moment sample supports FP16 precision only (`export PRECISION=fp16`).

## Run Using Docker

### Set up Docker Image
```bash
docker pull intel/generative-ai:pytorch-flex-gpu-stable-diffusion-inference
```

### Run Docker Image
The stable diffusion inference container includes scripts, model and libraries needed to run FP16 inference. To run the `run_model.sh` quickstart script using this container, you will need to provide an output directory where log files will be written.

```bash
#Optional
export PRECISION=fp16
export BATCH_SIZE=<batch size otherwise (default: 1)>

#Required
export OUTPUT_DIR=<path to output directory>
export SCRIPT=run_model.sh
export MULTI_TILE=False
export PLATFORM=ARC

IMAGE_NAME=intel/generative-ai:pytorch-flex-gpu-stable-diffusion-inference
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

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/flex-gpu)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
