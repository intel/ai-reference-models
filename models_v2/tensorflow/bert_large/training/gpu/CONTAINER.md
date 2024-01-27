# TensorFlow BERT Large Training 

## Description
This document has instructions for running BERT Large training with BF16 precision using Intel(R) Extension for TensorFlow on Intel® Data Center GPU Max Series.

## Datasets

Follow instructions [here](README.md#prepare-dataset) to download and prepare the dataset for training. Set the environment variable `DATA_DIR` to point to the dataset directory.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Runs BERT Large BF16 training on single and two tiles |

Requirements:
* Host has [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)
* Follow instructions to install GPU-compatible [driver](https://dgpu-docs.intel.com/driver/installation.html)
* Docker

## Docker pull Command
```
docker pull intel/language-modeling:tf-max-gpu-bert-large-training
```
The BERT Large training container includes scripts, models and libraries needed to run BF16 training.You wil need to volume mount the dataset directory and the output directory where log files will be generated. 

```bash
export DATA_DIR=<path to processed dataset>
export RESULTS_DIR=<path to output log files>
export MULTI_TILE=<provide True for multi-tile training and False for single tile training>
export DATATYPE=bf16

IMAGE_NAME=intel/language-modeling:tf-max-gpu-bert-large-training
DOCKER_ARGS="--rm -it"
SCRIPT= run_model.sh

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env DATA_DIR=${DATA_DIR} \
  --env RESULTS_DIR=${RESULTS_DIR} \
  --env MULTI_TILE=${MUTI_TILE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --env DATATYPE=${DATATYPE} \
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
Support for Intel® Extension for TensorFlow* is available at [Intel® AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
