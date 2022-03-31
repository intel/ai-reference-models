<!--- 0. Title -->
# Faster_RCNN Int8 inference

<!-- 10. Description -->
## Description

This document has instructions for running Faster_RCNN Int8 inference using
Intel Optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[faster_rcnn-int8-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/faster-rcnn-int8-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

The [COCO validation dataset](http://cocodataset.org) is used in the
Faster_RCNN quickstart scripts. The scripts require that the dataset
has been converted to the TF records format. See the
[COCO dataset](/datasets/coco/README.md) for instructions on downloading
and preprocessing the COCO validation dataset.

Set the `DATASET_DIR` to point to this directory when running Faster_RCNN.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [int8_inference.sh](int8_inference.sh) | Runs batch and online inference using the raw coco dataset |
| [int8_accuracy.sh](int8_accuracy.sh) | Runs inference and evaluates the model's accuracy with coco dataset in `coco_val.record` format |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
* [Kubernetes](#kubernetes)

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* git
* numactl
* wget
* [Protobuf Compilation](https://github.com/tensorflow/models/blob/v1.12.0/research/object_detection/g3doc/installation.md#protobuf-compilation)
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* Cython
* contextlib2
* jupyter
* lxml
* matplotlib
* pillow>=8.1.2
* pycocotools

Clone the [TensorFlow Model Garden](https://github.com/tensorflow/models)
repository using the specified tag, and save the path to the `TF_MODELS_DIR`
environment variable.
```
# Clone the TF models repo
git clone https://github.com/tensorflow/models.git
pushd models
git checkout tags/v1.12.0
export TF_MODELS_DIR=$(pwd)
popd
```

After installing the prerequisites, download and extract the model
package, which includes the pretrained model and scripts needed
to run inference. Set environment variables for the path to
your `DATASET_DIR` (where the coco TF records file is
located) and an `OUTPUT_DIR` where log files will be written, then run a
[quickstart script](#quick-start-scripts).

```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/faster-rcnn-int8-inference.tar.gz
tar -xzf faster_rcnn-int8-inference.tar.gz
cd faster_rcnn-int8-inference

export DATASET_DIR=<path to the coco dataset>
export OUTPUT_DIR=<directory where log files will be written>

./quickstart/<script name>.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
Faster_RCNN Int8 inference. To run one of the quickstart scripts 
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
  intel/object-detection:tf-1.15.2-faster-rcnn-int8-inference \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

