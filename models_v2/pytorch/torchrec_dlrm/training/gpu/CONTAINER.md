# PyTorch DLRM-v2 Training

## Description 
This document has instructions for running DLRM-v2 training using FP32,TF32,BF16 precisions on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
|`run_model.sh` | Runs DLRM-v2 multi-tile distributed training on single-node |

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html) 1550 |
| Drivers | GPU-compatible drivers need to be installed: Download [Driver](https://dgpu-docs.intel.com/driver/installation.html) |
| Software | Docker* |

## Datasets
Refer to the [link](README.md#prepare-dataset) to download and prepare the Criteo 1TB Click Logs dataset. Set `DATASET_DIR` variable to point to the dataset directory. 

## Docker pull Command

```bash
docker pull intel/recommendation:pytorch-max-gpu-dlrm-v2-training
```
The DLRM training container includes scripts,models,libraries needed to run TF32,FP32 and BF16 training. To run the quickstart script follow the instructions below. 

```bash
#optional
export PRECISION=<provide FP32,TF32 or BF16 otherwise, (default: BF16)>
export BATCH_SIZE=<provide batch size,otherwise (default: 65536)>

#required
export DATASET_DIR=<provide path to the Criteo Dataset directory>
export MULTI_TILE=True
export PLATFORM=Max
export OUTPUT_DIR=<path to output directory to view logs> 

DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/recommendation:pytorch-max-gpu-dlrm-training
SCRIPT=run_model.sh

docker run \
  --device=/dev/dri \
  --ipc=host \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env PRECISION=${PRECISION} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env PLATFORM=${PLATFORM} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume /dev/dri:/dev/dri \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
```
## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/max-gpu)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
