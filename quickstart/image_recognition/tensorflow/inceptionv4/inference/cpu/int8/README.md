<!--- 0. Title -->
# Inception V4 Int8 inference

<!-- 10. Description -->
## Description

This document has instructions for running Inception V4 Int8 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[inceptionv4-int8-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/inceptionv4-int8-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to this directory when running Inception V4.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`int8_online_inference.sh`](/quickstart/image_recognition/tensorflow/inceptionv4/inference/cpu/int8/int8_online_inference.sh) | Runs online inference (batch_size=1). |
| [`int8_batch_inference.sh`](/quickstart/image_recognition/tensorflow/inceptionv4/inference/cpu/int8/int8_batch_inference.sh) | Runs batch inference (batch_size=240). |
| [`int8_accuracy.sh`](/quickstart/image_recognition/tensorflow/inceptionv4/inference/cpu/int8/int8_accuracy.sh) | Measures the model accuracy (batch_size=100). |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package.
Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/inceptionv4-int8-inference.tar.gz
tar -xzf inceptionv4-int8-inference.tar.gz
cd inceptionv4-int8-inference

./quickstart/<script name>.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
Inception V4 Int8 inference. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory.

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/image-recognition:tf-latest-inceptionv4-int8-inference \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

