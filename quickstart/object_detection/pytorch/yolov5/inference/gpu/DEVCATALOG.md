# Running FP16 YOLOv5 inference on Intel® Data Center GPU Flex Series using Intel® Extension for PyTorch*

## Overview

This document has instructions for running YOLOv5 inference using Intel® Extension for PyTorch on Intel® Flex Series GPU.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Data Center GPU Flex Series 170 or 140  |
| Drivers | GPU-compatible drivers need to be installed:[Download Driver 647](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html)
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
variable for YOLOv5 (for example: `export IMAGE_FILE=/home/<user>/coco/val2017/000000581781.jpg`). In addition, we should also set the `size` environment to match the size of image.
(for example: `export size=416`)

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Inference with FP16 for specified batch size on Flex series 170 and 140 |

## Run Using Docker

### Set up Docker Image

```bash
docker pull intel/object-detection:pytorch-flex-gpu-yolov5-inference
```
### Run Docker Image
The Yolov5 inference container includes scripts,model and libraries need to run int8 inference. To run the `inference.sh` quickstart script using this container, the script uses dummy image data. The script downloads the YOLOv5 pre-trained model .You will need to provide an output directory where log files will be written. 

```bash
export IMAGE_NAME=intel/object-detection:pytorch-flex-gpu-yolov5-inference
export IMAGE_FILE=<path to coco dataset image>
export BATCH_SIZE=<inference batch size>
export NUM_ITERATIONS=<number of iterations>
export OUTPUT_DIR=<path to output directory>
export PRECISION=<provide fp16 precision>
export SCRIPT=quickstart/inference.sh
export GPU_TYPE=<provide either flex_170 or flex_140>
DOCKER_ARGS="--rm -it"

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env IMAGE_FILE=${IMAGE_FILE} \
  --env PRECSION=${PRECISION} \
  --env GPU_TYPE=${GPU_TYPE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${IMAGE_FILE}:${IMAGE_FILE} \
  ${DOCKER_ARGS} \
  ${IMAGE_NAME} \
  /bin/bash $SCRIPT
```
**Note:**  Add `--cap-add=SYS_NICE` to the `docker run` command for executing `batch_inference.sh` on Flex series 140.
## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/flex-gpu)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
