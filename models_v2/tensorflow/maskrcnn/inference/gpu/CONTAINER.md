# Running MaskRCNN Inference with FP16 on Intel® Data Center GPU Flex Series using Intel® Extension for TensorFlow*

## Overview

This document has instructions for running MaskRCNN inference using Intel® Extension for TensorFlow* with Intel® Data Center GPU Flex Series.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html) 170 or 140  |
| Drivers | GPU-compatible drivers need to be installed: [Download Driver](https://dgpu-docs.intel.com/driver/installation.html)
| Software | Docker* |

## Get Started

### Download Dataset

This repository provides scripts to download and extract the [COCO 2017 dataset](http://cocodataset.org/#download).

Download and pre-process the datasets using script `download_and_preprocess_coco.sh` provided [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN/dataset). Set the `DATASET_DIR` to point to the TF records directory when running MaskRCNN.

### Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `run_model.sh` | Runs batch and online inference for FP16 precision on Flex series 170 and 140 |

## Run Using Docker

### Set up Docker Image

```
docker pull intel/image-segmentation:tf-flex-gpu-maskrcnn-inference
```

### Run Docker Image
The MaskRCNN inference container includes scripts,model and libraries needed to run FP16 batch and online inference. To run the inference script using this container, you'll need to provide volume mounts for the COCO processed dataset.You will also need to provide an output directory where log files will be written. 

```bash
#Optional
export BATCH_SIZE=<provide batch size. default batch for batch inference is 16>

#Required
export PRECISION=<provide precision,supports float16>
export OUTPUT_DIR=<path to output directory>
export DATASET_DIR=<path to the preprocessed COCO dataset>
export GPU_TYPE=<provide either flex_170 or flex_140>

IMAGE_NAME=intel/image-segmentation:tf-flex-gpu-maskrcnn-inference
SCRIPT=run_model.sh

docker run -it \
  --device=/dev/dri \
  --ipc=host \
  --privileged \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env PRECSION=${PRECISION} \
  --env GPU_TYPE=${GPU_TYPE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --rm -it \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
  ```
**Note:**  Add `--cap-add=SYS_NICE` to the `docker run` command for executing `run_model.sh` on Flex series 140.

## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/flex-gpu)

## Support
Support for Intel® Extension for TensorFlow* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
