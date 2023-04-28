# Running SSD-MobileNet Inference on Intel® Data Center GPU Flex Series using Intel® Extension for TensorFlow*

## Overview

This document has instructions for running SSD-MobileNet inference using Intel®  Extension for TensorFlow* with Intel® Data Center GPU Flex Series.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Data Center GPU Flex Series  |
| Drivers | GPU-compatible drivers need to be installed:[Download Driver 555](https://dgpu-docs.intel.com/releases/stable_555_20230124.html#ubuntu-22-04)
| Software | Docker* Installed |

## Get Started

### Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `flex_multi_card_online_inference` | Runs online inference for int8 precision |
| `flex_multi_card_batch_inference` | Runs batch inference for int8 precision |

## Run Using Docker

### Set up Docker Image

```
docker pull intel/object-detection:tf-flex-gpu-ssd-mobilenet-multi-card-inference
```
### Run Docker Image
The SSD-MobileNet inference container includes scripts,model and libraries need to run int8 inference. Dummy dataset will be used for run inference. For example,to run `flex_multi_card_batch_inference` quickstart scripts using this you will need to provide an output directory where log files will be writte
```
export PRECISION=int8
export OUTPUT_DIR=<path to output directory>
export BATCH_SIZE=<enter batch size. Default is 256>
IMAGE_NAME=intel/object-detection:tf-flex-gpu-ssd-mobilenet-multi-card-inference
SCRIPT=quickstart/flex_multi_card_batch_inference.sh
DOCKER_ARGS="--rm -it"

VIDEO=$(getent group video | sed -E 's,^video:[^:]*:([^:]*):.*$,\1,')
RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,')

test -z "$RENDER" || RENDER_GROUP="--group-add ${RENDER}"

docker run \
  --group-add ${VIDEO} \
  ${RENDER_GROUP} \
  --device=/dev/dri \
  --ipc=host \
  --cap-add=SYS_NICE \
  --env PRECISION=${PRECISION} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_ARGS} \
  ${IMAGE_NAME} \
  /bin/bash $SCRIPT
```

## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/dockerfiles/model_containers)

## Support
Support for Intel® Extension for TensorFlow* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for TensorFlow* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.