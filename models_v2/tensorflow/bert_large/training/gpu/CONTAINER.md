# TensorFlow BERT Large Training 

## Description
This document has instructions for running BERT Large training with BF16,FP32 and TF32 precisions using Intel(R) Extension for TensorFlow on Intel® Data Center GPU Max Series.

## Datasets

Follow [instructions](README.md#prepare-dataset) to download and prepare the dataset for training. Set the environment variable `DATA_DIR` to point to the dataset directory.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Runs BERT Large BF16,FP32 and TF32 training on single or multiple GPU devices |

Requirements:
* Host has [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)
* Follow instructions to install GPU-compatible [driver](https://dgpu-docs.intel.com/driver/installation.html)
* Docker

## Docker pull Command
```
docker pull intel/language-modeling:tf-max-gpu-bert-large-training
```
The BERT Large training container includes scripts, models and libraries needed to run BF16/FP32/TF32 training.You wil need to volume mount the dataset directory and the output directory where log files will be generated. 

```bash
export DATA_DIR=<path to processed dataset>
export RESULTS_DIR=<path to output log files>
export DATATYPE=<provide bf16,fp32 or tf32 precision>
export MULTI_TILE=<provide True for multi-tile GPU such as Max 1550, and False for single-tile GPU such as Max 1100>
export NUM_DEVICES=<provide the number of GPU devices used for training. It must be equal to or smaller than the number of GPU devices attached to each node. For GPU with 2 tiles, such as Max 1550 GPU, the number of GPU devices in each node is 2 times the number of GPUs, so it can be set as <=16 for a node with 8 Max 1550 GPUs. While for GPU with single tile, such as Max 1100 GPU, the number of GPU devices available in each node is the same as number of GPUs, so it can be set as <=8 for a node with 8 Max 1100 GPUs.>

IMAGE_NAME=intel/language-modeling:tf-max-gpu-bert-large-training
DOCKER_ARGS="--rm -it"
SCRIPT=run_model.sh

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env DATA_DIR=${DATA_DIR} \
  --env RESULTS_DIR=${RESULTS_DIR} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --env DATATYPE=${DATATYPE} \
  --env NUM_DEVICES=${NUM_DEVICES} \
  --volume ${RESULTS_DIR}:${RESULTS_DIR} \
  --volume ${DATA_DIR}:${DATA_DIR} \
  --volume /dev/dri:/dev/dri \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
  ```

## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/max-gpu)

## Support
Support for Intel® Extension for TensorFlow* is available at [Intel® AI Tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html). Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
