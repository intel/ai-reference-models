<!--- 0. Title -->
# ResNet50 v1.5 Int8 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 v1.5 Int8 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[resnet50v1-5-int8-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/resnet50v1-5-int8-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`int8_online_inference.sh`](int8_online_inference.sh) | Runs online inference (batch_size=1). |
| [`int8_batch_inference.sh`](int8_batch_inference.sh) | Runs batch inference (batch_size=128). |
| [`int8_accuracy.sh`](int8_accuracy.sh) | Measures the model accuracy (batch_size=100). |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow==2.3.0](https://pypi.org/project/intel-tensorflow/)
* numactl

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

For accuracy, `DATASET_DIR` is required to be set. For inference,
just to evaluate performance on sythetic data, the `DATASET_DIR` is not needed.
Otherwise `DATASET_DIR` needs to be set:

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/resnet50v1-5-int8-inference.tar.gz
tar -xzf resnet50v1-5-int8-inference.tar.gz
cd resnet50v1-5-int8-inference

quickstart/<script name>.sh
```

<!--- 60. Docker -->
## Docker

The model container `intel/image-recognition:tf-2.3.0-imz-2.2.0-resnet50v1-5-int8-inference` includes the scripts and libraries needed to run
ResNet50 v1.5 Int8 inference. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory.

For accuracy, `DATASET_DIR` is required to be set. For inference,
just to evaluate performance on synthetic data, the `DATASET_DIR` is not needed.
Otherwise `DATASET_DIR` needs to be set. Add or remove `DATASET_DIR` environment
variable and volume mount accordingly in following command:


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
  intel/image-recognition:tf-2.3.0-imz-2.2.0-resnet50v1-5-int8-inference \
  /bin/bash quickstart/<script name>.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

