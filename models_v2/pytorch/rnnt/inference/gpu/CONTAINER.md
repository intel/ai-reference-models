# Running RNN-T Inference on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch*

## Description 
This document has instructions for running RNN-T Inference using Intel® Extension for PyTorch on Intel®Max Series GPU. 

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)  |
| Drivers | GPU-compatible drivers need to be installed: Download [Driver](https://dgpu-docs.intel.com/driver/installation.html) |
| Software | Docker* |

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Runs FP16,BF16 and FP32 inference on single tile and two tiles|

## Datasets and Pre-trained Models

Refer to [instructions](./README.md#prepare-dataset) to download and pre-process LibriSpeech dataset. Refer to the instructions to download the pre-trained model.

Set the `DATASET_DIR` and `WEIGHT_DIR` environment variables to point to the dataset and model directories respectively. 

### Docker pull command:

```
docker pull intel/speech-recognition:pytorch-max-gpu-rnnt-inference
```
The RNNT inference container includes scripts, model and libraries needed to run FP16,BF16 and FP32 inference. To run the `run_model.sh` quickstart script using this container, you'll need to set the environment variable and provide volume mounts for the Dataset and Pre-trained model. You will need to provide an output directory where log files will be written. 

```bash
#Optional
export PRECISION=<provide either BF16,FP16,FP32 otherwise (default: BF16)>
export BATCH_SIZE=<provide batch size, otherwise (default: 512)>

#Required
export DATASET_DIR=<provide path to processed dataset>
export WEIGHT_DIR=<provide path to pre-trained model directory>
export PLATFORM=Max
export MULTI_TILE=<provide True of False to enable/disable multi-tile inference>
export OUTPUT_DIR=<path to output logs directory>

IMAGE_NAME=intel/speech-recognition:pytorch-max-gpu-rnnt-inference
DOCKER_ARGS="--rm -it"
SCRIPT=run_model.sh

docker run \
  --device=/dev/dri \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env WEIGHT_DIR=${WEIGHT_DIR} \
  --env PRECISION=${PRECISION} \
  --env PLATFORM=${PLATFORM} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${WEIGHT_DIR}:${WEIGHT_DIR} \
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
