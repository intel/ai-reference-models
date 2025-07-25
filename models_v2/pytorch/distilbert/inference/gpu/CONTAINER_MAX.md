# Running DistilBERT inference on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch*

## Overview

This document has instructions for running DistilBERT inference inference using Intel® Extension for PyTorch on Intel® Max Series GPU.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html) |
| Drivers | GPU-compatible drivers need to be installed: Download [Driver](https://dgpu-docs.intel.com/driver/installation.html) |
| Software | Docker* |

# Datasets

Refer to [instructions](README.md#dataset) to download and prepare the dataset. Set `DATASET_DIR` to point to the dataset directory. 

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Inference with batch size 32 on Max series |

## Run Using Docker

### Set up Docker Image

```
docker pull intel/language-modeling:pytorch-max-gpu-distilbert-inference
```

### Run Docker Image
The distilbert inference container includes scripts, model and libraries needed to run FP16,BF16 and FP32 inference. To run the `run_model.sh` quickstart script using this container, you will need to provide an output directory where log files will be written.

```bash
#Optional 
export PRECISION=<provide FP32, BF16 or FP16 otherwise (default:FP16)>
export BATCH_SIZE=<provide batch size otherwise (default:32)>
export NUM_ITERATIONS=<provide num_iterations otherwise (default:300)>

#Required
export PLATFORM=Max
export MULTI_TILE=True
export OUTPUT_DIR=<path to output directory>
export SCRIPT=run_model.sh
export DATASET_DIR=<path to dataset directory>

IMAGE=intel/language-modeling:pytorch-max-gpu-distilbert-inference
DOCKER_ARGS="--rm -it"

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env PRECISION=${PRECISION} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env PLATFORM=${PLATFORM} \
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

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/max-gpu)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
