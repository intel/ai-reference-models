<!--- 0. Title -->
# Faster R-CNN FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running Faster R-CNN FP32 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[faster-rcnn-fp32-inference.tar.gz](https://ubit-artifactory-or.intel.com/artifactory/aipg-local/aipg-tf/modular-zoo-model-package-generator/239/faster-rcnn-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

The [COCO validation dataset](http://cocodataset.org) is used in the
Faster R-CNN quickstart scripts. The scripts require that the dataset
has been converted to the TF records format. See the
[COCO dataset](/datasets/coco/README.md) for instructions on downloading
and preprocessing the COCO validation dataset.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp32_inference.sh](fp32_inference.sh) | Runs batch and online inference using the coco dataset |
| [fp32_accuracy.sh](fp32_accuracy.sh) | Runs inference and evaluates the model's accuracy |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python
* git
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/)
* numactl
* Cython
* contextlib2
* pillow>=7.1.0
* lxml
* jupyter
* matplotlib
* pycocotools
* [Protobuf Compilation](https://github.com/tensorflow/models/blob/v1.12.0/research/object_detection/g3doc/installation.md#protobuf-compilation)

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

Download and extract the model package, which includes the pretrained
model and scripts needed to run inference. Set environment variables
for the path to your `DATASET_DIR` (where the coco TF records file is
located) and an `OUTPUT_DIR` where log files will be written, then run a
[quickstart script](#quick-start-scripts).
```
wget https://ubit-artifactory-or.intel.com/artifactory/aipg-local/aipg-tf/modular-zoo-model-package-generator/239/faster-rcnn-fp32-inference.tar.gz
tar -xzf faster-rcnn-fp32-inference.tar.gz
cd faster-rcnn-fp32-inference

export DATASET_DIR=<path to the coco dataset>
export OUTPUT_DIR=<directory where log files will be written>

quickstart/<script name>.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
Faster R-CNN FP32 inference and the pretrained model. To run one of the quickstart scripts
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
  amr-registry.caas.intel.com/aipg-tf/model-zoo-ci:239-ci-build-1.15.2-object-detection-faster-rcnn-fp32-inference \
  /bin/bash quickstart/<script name>.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

