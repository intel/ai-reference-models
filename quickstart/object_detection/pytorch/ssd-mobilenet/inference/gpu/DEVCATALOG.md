# Running SSD-MobileNetv1 Inference on Intel® Data Center GPU Flex Series using Intel® Extension for PyTorch*

## Overview

This document has instructions for running SSD-Mobilenetv1 inference using Intel® Extension for PyTorch on Intel® Flex Series GPU.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | Intel® Data Center GPU Flex Series  |
| Drivers | GPU-compatible drivers need to be installed: [Download Driver 602](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-dc.html#step-1-add-package-repository)
| Software | Docker* Installed |

## Download Datasets

The [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) validation dataset is used.

Download and extract the VOC2007 dataset from http://host.robots.ox.ac.uk/pascal/VOC/voc2007/,
After extracting the data, your folder structure should look something like this:

```
VOC2007
├── Annotations
│   ├── 000038.xml    
│   ├── 000724.xml
│   ├── 001440.xml
│   └── ...
├── ImageSets
│   ├── Layout    
│   ├── Main
│   └── Segmentation
├── SegmentationClass
│   ├── 005797.png   
│   ├── 007415.png 
│   ├── 006581.png 
│   └── ...
├── SegmentationObject
│   ├── 005797.png    
│   ├── 006581.png
│   ├── 007415.png
│   └── ...
└── JPEGImages
    ├── 002832.jpg    
    ├── 003558.jpg
    ├── 004262.jpg
    └── ...
```
The folder should be set as the `DATASET_DIR`
(for example: `export DATASET_DIR=/home/<user>/VOC2007`).

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_with_dummy_data.sh` | Inference with dummy data, for int8 blocked channel first on Flex series 170 |
| `flex_multi_card_batch_inference.sh` | Inference with dummy data,for int8 and given batch size blocked channel first on Flex series 140 |
| `flex_multi_card_online_inference.sh` | Online Inference with dummy data,for int8 blocked channel first on Flex series 140 | 

## Run Using Docker

### Set up Docker Image

```
docker pull intel/object-detection:pytorch-flex-gpu-ssd-mobilenet-inference
```
### Run Docker Image
The SSD-MobileNet inference container includes scripts,model and libraries need to run int8 inference. To run the `flex_multi_card_batch_inference.sh` quickstart script using this container,you will need to provide an output directory where log files will be written.The script by default runs inference on dummy data. The script also performs online INT8 calibration for which the VOC2007 dataset is required to be provided. The pre-trained model will be downloaded by the script.

**Note:** The Default batch size for Flex series 140 is 256 for batch inference and 1024 for Flex series 170. Additionally, add `--cap-add=SYS_NICE` to the `docker run` command for executing `flex_multi_card_online_inference.sh` and `flex_multi_card_batch_inference.sh` on Flex series 140.

```
export PRECISION=int8
export OUTPUT_DIR=<path to output directory>
export DATASET_DIR=<path to the preprocessed voc2007 dataset>
export PRETRAINED_MODEL=<path to the pretrained model folder. The code downloads the model if this folder is empty>
export SCRIPT=quickstart/inference_with_dummy_data.sh
export BATCH_SIZE=<inference batch size>
export NUM_ITERATIONS=<enter number of iterations>
export label=/workspace/pytorch-flex-series-ssd-mobilenet-inference/labels/voc-model-labels.txt

DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/object-detection:pytorch-flex-gpu-ssd-mobilenet-inference 

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env label=${label} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
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
