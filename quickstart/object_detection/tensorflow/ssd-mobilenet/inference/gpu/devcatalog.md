# SSD-MobileNet Inference

## Description

This document has instructions for running SSD-MobileNet inference using
Intel(R) Extension for TensorFlow* with Intel(R) Data Center GPU Flex Series.

## Datasets

Download and preprocess the COCO dataset using the [instructions here](https://github.com/IntelAI/models/blob/master/datasets/coco/README.md).
After running the conversion script you should have a directory with the
COCO dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running SSD-MobileNet.

## Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `online_inference` | Runs online inference for int8 precision | 
| `batch_inference` | Runs batch inference for int8 precision |
| `accuracy` | Measures the model accuracy for int8 precision |

## Docker

Requirements:
* Host machine has Intel(R) Data Center GPU Flex Series
* Follow instructions to install GPU-compatible driver [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html)
* Docker

### Docker pull command:

```
docker pull intel/object-detection:tf-atsm-gpu-ssd-mobilenet-inference
```

The SSD-MobileNet inference container includes scripts,model and libraries need to run int8 inference. To run the inference quickstart scripts using this container, you'll need to provide volume mounts for the COCO dataset for running `accuracy.sh` script. For `online_inference.sh` and `batch_inference.sh` dummy dataset will be used. You will need to provide an output directory where log files will be written. 

```
export PRECISION=int8
export OUTPUT_DIR=<path to output directory>
export DATASET_DIR=<path to the preprocessed coco dataset>
IMAGE_NAME=intel/object-detection:tf-atsm-gpu-ssd-mobilenet-inference

VIDEO=$(getent group video | sed -E 's,^video:[^:]*:([^:]*):.*$,\1,')
RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,')

docker run \
  --group-add ${VIDEO} \
  ${RENDER_GROUP} \
  --device=/dev/dri \
  --ipc=host \
  --privileged \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --rm --it \
  $IMAGE_NAME \
  /bin/bash quickstart/<script name>.sh
```

## Documentation and Sources

**Get Started**

[Docker* Repository](https://hub.docker.com/r/intel/image-recognition)

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
