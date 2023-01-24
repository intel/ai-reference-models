<!--- 0. Title -->
# ResNet101 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet101 inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to this directory when running ResNet101.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`online_inference.sh`](/quickstart/image_recognition/tensorflow/resnet101/inference/cpu/online_inference.sh) | Runs online inference (batch_size=1). |
| [`batch_inference.sh`](/quickstart/image_recognition/tensorflow/resnet101/inference/cpu/batch_inference.sh) | Runs batch inference (batch_size=128). |
| [`accuracy.sh`](/quickstart/image_recognition/tensorflow/resnet101/inference/cpu/accuracy.sh) | Measures the model accuracy (batch_size=100). |

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
ResNet101 inference. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory.

```
DATASET_DIR=<path to the dataset>
PRECISION=<set the precision "int8" or "fp32">
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION}
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/image-recognition:tf-latest-resnet101-inference \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

