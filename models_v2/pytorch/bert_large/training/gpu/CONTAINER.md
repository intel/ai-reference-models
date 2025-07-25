# Running BERT Large Training on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch*

## Description
This document has instructions for running BERT Large training using BF16,TF32 and FP32 precisions using Intel® Extension for PyTorch on Intel Max Series GPU. 

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Runs BERT Large training on single or multiple GPU devices |

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)  |
| Drivers | GPU-compatible drivers need to be installed: Download [Driver](https://dgpu-docs.intel.com/driver/installation.html) |
| Software | Docker* |

## Datasets
### Download and Extract the Dataset
[Refer](README.md#dataset) to download and prepare dataset for training. Set `DATASET_DIR` to point to the dataset directory. 

## Docker pull Command
```
docker pull intel/language-modeling:pytorch-max-gpu-bert-large-training
```

The BERT Large training container includes scripts, models,libraries needed to run BF16,TF32 and FP32 training. To run the `run_model.sh` script, you will need to provide an output directory where log files will be written. 

```bash
#optional
export PRECISION=<provide FP32,TF32 or BF16 otherwise, (default: BF16)>
export NUM_ITERATIONS=<provide num_iterations,otherwise (default: 10)>
export BATCH_SIZE=<provide batch size,otherwise (default: 16)>

#required
export OUTPUT_DIR=<path to output directory>
export DATASET_DIR=<path to dataset directory>
export MULTI_TILE=<provide True for multi-tile GPU such as Max 1550, and False for single-tile GPU such as Max 1100>
export NUM_DEVICES=<provide the number of GPU devices used for training. It must be equal to or smaller than the number of GPU devices attached to each node. For GPU with 2 tiles, such as Max 1550 GPU, the number of GPU devices in each node is 2 times the number of GPUs, so it can be set as <=16 for a node with 8 Max 1550 GPUs. While for GPU with single tile, such as Max 1100 GPU, the number of GPU devices available in each node is the same as number of GPUs, so it can be set as <=8 for a node with 8 Max 1100 GPUs.>
export PLATFORM=Max

DOCKER_ARGS="--rm --init -it"
IMAGE_NAME=intel/language-modeling:pytorch-max-gpu-bert-large-training
SCRIPT=run_model.sh

docker run \
  --device=/dev/dri \
  --ipc host \
  --env DATASET_DIR=${DATASET_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env PRECISION=${PRECISION} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env PLATFORM=${PLATFORM} \
  --env NUM_DEVICES=${NUM_DEVICES} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume /dev/dri:/dev/dri/ \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
  ```

## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/max-gpu)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html). Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
