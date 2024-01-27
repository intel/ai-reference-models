# TensorFlow MaskRCNN Training

## Description

This document has instructions for running MaskRCNN training with BFloat16 precision using Intel® Extension for TensorFlow on Intel® Data Center GPU Max Series.

## Datasets

Download and preprocess the COCO 2017 dataset using the [instructions here](./README.md). After running the conversion script you should have a directory with the COCO 2017 dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running MaskRCNN. 

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Runs MaskRCNN BF16 training on single and two tiles |

Requirements:
* Host has [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)
* Follow instructions to install GPU-compatible [drivers](https://dgpu-docs.intel.com/driver/installation.html)
* Docker

## Docker pull command:

```bash
docker pull intel/image-segmentation:tf-max-gpu-maskrcnn-training
```
To MaskRCNN training container includes scripts, models and libraries needed to run BFloat16 training. To run the `run_model.sh` quickstart script using this container, you are required to volume mount the pre-processed COCO 2017 dataset to run the script. You will also need to provide an output directory to store logs.

```bash
#Optional
export BATCH_SIZE=<provide batch size. Default is 4>

#Required
export DATASET_DIR=<path to pre-processed COCO 2017 dataset>
export MULTI_TILE=<specify True for Multi-tile training and False for single-tile training>
export PRECISION=bfloat16
export OUTPUT_DIR=<path to output directory>

IMAGE_NAME=intel/image-segmentation:tf-max-gpu-maskrcnn-training

if [[ ${MULTI_TILE} == "False" ]]; then
    SCRIPT="mpirun -np 1 -prepend-rank -ppn 1 bash run_model.sh"
else 
    SCRIPT="bash run_model.sh"
fi

DOCKER_ARGS="--rm --init -it"
docker run \
--device /dev/dri \
--env BATCH_SIZE=${BATCH_SIZE} \
--env MULTI_TILE=${MULTI_TILE} \
--env DATASET_DIR=${DATASET_DIR} \
--env PRECISION=${PRECISION} \
--env OUTPUT_DIR=${OUTPUT_DIR} \
--volume /dev/dri:/dev/dri \
--volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
--volume ${DATASET_DIR}:${DATASET_DIR} \
${DOCKER_ARGS} \
${IMAGE_NAME} \
$SCRIPT
```
## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/max-gpu)

## Support
Support for Intel® Extension for TensorFlow* is available at [Intel® AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
