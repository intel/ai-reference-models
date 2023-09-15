# Running Stable Diffusion Inference with FP32 and FP16 on Intel® Data Center GPU Flex Series using Intel® Extension for TensorFlow*

## Overview

This document has instructions for running Stable Diffusion inference using Intel® Extension for TensorFlow* with Intel® Data Center GPU Flex Series.


## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Data Center GPU Flex Series  |
| Drivers | GPU-compatible drivers need to be installed: [Download Driver 647](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html)
| Software | Docker* Installed |

## Get Started

### Dataset

For accuracy measurement, please download the reference data file `img_arrays_for_acc.txt` from the link [here](https://github.com/intel/intel-extension-for-tensorflow/tree/main/examples/stable_diffussion_inference/nv_results). Set the `REFERENCE_RESULT_FILE` to point to the file.

### Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `online_inference` | Runs online inference for FP32,FP16 precisions on Flex series 170 |
| `accuracy` | Runs batch inference for FP16 precision on Flex series 170 |

## Run Using Docker

### Set up Docker Image

```bash
docker pull intel/generative-ai:tf-flex-gpu-stable-diffusion-inference
```

### Run Docker Image

The Stable Diffusion inference container contains scripts,models and libraries needed to run FP32 and FP16 inference. You will need to provide an output directory where log files will be written. To run the accuracy test, additionally, you will have to download and volume mount the reference data file. 

```bash
export IMAGE_NAME=intel/generative-ai:tf-flex-gpu-stable-diffusion-inference
export BATCH_SIZE=1
export PRECISION=< provide fp32 or fp16 as precision input >
export OUTPUT_DIR=<path to output directory>
export REFERENCE_RESULT_FILE=<path to reference results file only for accuracy >

docker run \
  --device=/dev/dri \
  --ipc=host \
  --privileged \
  --env PRECISION=${PRECISION} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env REFERENCE_RESULT_FILE=${REFERENCE_RESULT_FILE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${REFERENCE_RESULT_FILE}:${REFERENCE_RESULT_FILE} \
  --rm -it \
  $IMAGE_NAME \
  /bin/bash quickstart/<script_name>.sh
```

## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/flex-gpu)

## Support
Support for Intel® Extension for TensorFlow* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
