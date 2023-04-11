# Running YOLOv4 inference on Intel® Data Center GPU Flex Series using Intel® Extension for PyTorch*


## Overview

This document has instructions for running YOLOv4 inference using Intel® Extension for PyTorch on Intel® Flex Series GPU.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Data Center GPU Flex Series  |
| Drivers | GPU-compatible drivers need to be installed:[Download Driver 555](https://dgpu-docs.intel.com/releases/stable_555_20230124.html#ubuntu-22-04)
| Software | Docker* Installed |

## Pretrained Model

You need to download pretrained weights from: yolov4.pth(https://pan.baidu.com/s/1ZroDvoGScDgtE1ja_QqJVw Extraction code:xrq9) or yolov4.pth(https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ) to the any directory of your choice, and set environment `PRETRAINED_MODEL`.

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `flex_multi_card_batch_inference.sh` | Inference with int8 for specified precision(int8) and batch size on dummy data |
| `flex_multi_card_online_inference.sh` | Online Inference with int8 for specified precision(int8) on dummy data |
## Run Using Docker

### Set up Docker Image

```
docker pull intel/object-detection:pytorch-flex-gpu-yolov4-multi-card-inference
```
### Run Docker Image
The Yolov4 inference container includes scripts,model and libraries need to run int8 inference. To run the `inference_with_dummy_data.sh` quickstart script using this container, the script uses dummy data. The script also performs online INT8 Calibration on the provided pre-trained model.You will need to provide an output directory where log files will be written. 

```
export PRECISION=int8
export OUTPUT_DIR=<path to output directory>
export SCRIPT=quickstart/flex_multi_card_batch_inference.sh
export PRETRAINED_MODEL=<path to downloaded yolov4 model>
export BATCH_SIZE=<enter batch size. Default is 64>
export NUM_ITERATIONS=<enter number of iterations. Default is 500>

IMAGE_NAME=intel/object-detection:pytorch-flex-gpu-yolov4-multi-card-inference
DOCKER_ARGS=${DOCKER_ARGS:---rm -it}
VIDEO=$(getent group video | sed -E 's,^video:[^:]*:([^:]*):.*$,\1,')
RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,')

test -z "$RENDER" || RENDER_GROUP="--group-add ${RENDER}"

docker run \
  --group-add ${VIDEO} \
  ${RENDER_GROUP} \
  --device=/dev/dri \
  --ipc=host \
  --cap-add=SYS_NICE \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env PRETRAINED_MODEL=${PRETRAINED_MODEL} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${PRETRAINED_MODEL}:${PRETRAINED_MODEL} \
  ${DOCKER_ARGS} \
  ${IMAGE_NAME} \
  /bin/bash $SCRIPT
  ```

## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/dockerfiles/model_containers)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.