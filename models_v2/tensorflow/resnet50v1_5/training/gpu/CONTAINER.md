# TensorFlow ResNet50_v1.5 Training

## Description

This document has instructions for running ResNet50 v1.5 training with BFloat16 precision using Intel® Extension for TensorFlow on Intel® Data Center GPU Max Series.

## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md). After running the conversion script you should have a directory with the ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Runs ResNet50 v1.5 BF16 training on single and two tiles |

Requirements:
* Host has [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)
* Follow instructions to install GPU-compatible [drivers](https://dgpu-docs.intel.com/driver/installation.html)
* Docker

### Docker pull command:
```
docker pull intel/image-recognition:tf-max-gpu-resnet50v1-5-training
```
The ResNet50 v1.5 training container includes scripts, models and libraries needed to run BFloat16 training. The script `run_model.sh` requires configuration files as input. For single-tile training, specify the configuration files present in `configure` folder as the `CONFIG_FILE` environment variable. For Multi-tile training, specify the configuration files in `hvd_configure` folder. You will also need to provide an output directory to store logs. To use Imagenet dataset, you will need to volume mount the dataset. For dummy data, use corresponding configuration files and volume mount of ImageNet dataset is not required and hence `DATASET_DIR` volume mounts are not required.

The following example shows how to run BF16 training using real data. 

```bash
#Optional
export DATASET_DIR=<path to pre-processed ImageNet datasets>

#Required
export CONFIG_FILE=/workspace/tf-max-series-resnet50v1-5-training/models/hvd_configure/itex_bf16_lars.yaml
export MULTI_TILE=<specify True for Multi-tile training and False for single-tile training>
export SCRIPT=run_model.sh
export OUTPUT_DIR=<provide path to output logs directory>

DOCKER_ARGS="--rm --init -it"
IMAGE_NAME=intel/image-recognition:tf-max-gpu-resnet50v1-5-training

docker run \
  --device=/dev/dri \
  --ipc=host \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env CONFIG_FILE=${CONFIG_FILE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
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
