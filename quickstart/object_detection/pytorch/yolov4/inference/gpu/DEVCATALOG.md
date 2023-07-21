# Running YOLOv4 inference on Intel® Data Center GPU Flex Series using Intel® Extension for PyTorch*


## Overview

This document has instructions for running YOLOv4 inference using Intel® Extension for PyTorch on Intel® Flex Series GPU.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Data Center GPU Flex Series  |
| Drivers | GPU-compatible drivers need to be installed:[Download Driver 602](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-dc.html#step-1-add-package-repository)
| Software | Docker* Installed |

## Dataset

Download and extract the 2017 training/validation images and annotations from the [COCO dataset website](https://cocodataset.org/#download) to a `coco` folder and unzip the files. After extracting the zip files, your dataset directory structure should look something like this:
```
coco
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017
│   ├── 000000454854.jpg
│   ├── 000000137045.jpg
│   ├── 000000129582.jpg
│   └── ...
└── val2017
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    ├── 000000000632.jpg
    └── ...
```
The parent of the `annotations`, `train2017`, and `val2017` directory (in this example `coco`) is the directory that should be used when setting the `IMAGE_FILE` environment
variable for YOLOv4 (for example: `export IMAGE_FILE=/home/<user>/coco/val2017/000000581781.jpg`). In addition, we should also set the `size` environment to match the size of image.
(for example: `export size=416`)

## Pretrained Model

You need to download pretrained weights from: yolov4.pth(https://pan.baidu.com/s/1ZroDvoGScDgtE1ja_QqJVw Extraction code:xrq9) or yolov4.pth(https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ) to the any directory of your choice, and set environment `PRETRAINED_MODEL`.

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_block_format.sh` | Inference with int8 for specified precision(int8) and batch size on dummy data on Flex series 170 |
| `flex_multi_card_batch_inference.sh` | Inference with int8 for specified precision(int8) and batch size on dummy data on Flex series 140 |
| `flex_multi_card_online_inference.sh` | Online Inference with int8 for specified precision(int8) on dummy data on Flex series 140 |
## Run Using Docker

### Set up Docker Image

```
docker pull intel/object-detection:pytorch-flex-gpu-yolov4-inference
```
### Run Docker Image
The Yolov4 inference container includes scripts,model and libraries need to run int8 inference. To run the `inference_block_format.sh` quickstart script using this container, the script uses dummy image data. The script also performs online INT8 Calibration on the provided pre-trained model.You will need to provide an output directory where log files will be written. 

**Note:** The default batch size of Flex series 170 is 256 and the batch size for Flex series 140 is 64. Add `--cap-add=SYS_NICE` to the `docker run` command for executing `flex_multi_card_online_inference.sh` and `flex_multi_card_batch_inference.sh` on Flex series 140.

```
export PRECISION=int8
export OUTPUT_DIR=<path to output directory>
export SCRIPT=quickstart/inference_block_format.sh
export PRETRAINED_MODEL=<path to downloaded yolov4 model>
export IMAGE_FILE=<path to coco dataset image>
export BATCH_SIZE=<inference batch size>
export NUM_ITERATIONS=<number of iterations>
IMAGE_NAME=intel/object-detection:pytorch-flex-gpu-yolov4-inference

DOCKER_ARGS="--rm -it"

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env IMAGE_FILE=${IMAGE_FILE} \
  --env PRETRAINED_MODEL=${PRETRAINED_MODEL} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${IMAGE_FILE}:${IMAGE_FILE} \
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
