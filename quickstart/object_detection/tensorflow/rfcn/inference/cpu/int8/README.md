<!--- 0. Title -->
# RFCN Int8 inference

<!-- 10. Description -->
## Description

This document has instructions for running RFCN Int8 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[rfcn-int8-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/rfcn-int8-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

The [COCO validation dataset](http://cocodataset.org) is used in these
RFCN quickstart scripts. The inference quickstart scripts use raw images,
and the accuracy quickstart scripts require the dataset to be converted
into the TF records format.
See the [COCO dataset](/datasets/coco/README.md) for instructions on
downloading and preprocessing the COCO validation dataset.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`int8_inference.sh`](/quickstart/object_detection/tensorflow/rfcn/inference/cpu/int8/int8_inference.sh) | Runs inference on a directory of raw images for 500 steps and outputs performance metrics. |
| [`int8_accuracy.sh`](/quickstart/object_detection/tensorflow/rfcn/inference/cpu/int8/int8_accuracy.sh) | Processes the TF records to run inference and check accuracy on the results. |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* RFCN uses the object detection code from the
[TensorFlow Model Garden](https://github.com/tensorflow/models). Clone this repo with the SHA specified
below and apply the patch from the RFCN Int8 inference model package to run with TF2.

```
git clone https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b
git apply ../rfcn-int8-inference/models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch
```

You must also install the dependencies and run the protobuf compilation described in the
[object detection installation instructions](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
from the [TensorFlow Model Garden](https://github.com/tensorflow/models) repository.


After installing the prerequisites, download and untar the model package.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/rfcn-int8-inference.tar.gz
tar -xzf rfcn-int8-inference.tar.gz
```

Set environment variables for the TensorFlow Model Garden directory to `TF_MODELS_DIR`, the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

To run inference with performance metrics:
```
DATASET_DIR=<path to the coco val2017 raw image directory (ex: /home/user/coco_dataset/val2017)>
OUTPUT_DIR=<directory where log files will be written>
TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>

cd rfcn-int8-inference.tar.gz
./quickstart/int8_inference.sh
```

To get accuracy metrics:
```
DATASET_DIR=<path to the COCO validation TF record file (ex: /home/user/coco_output/coco_val.record)>
OUTPUT_DIR=<directory where log files will be written>
TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>

cd rfcn-int8-inference.tar.gz
./quickstart/int8_accuracy.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
RFCN Int8 inference. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory.

To run inference with performance metrics:
```
DATASET_DIR=<path to the coco val2017 raw image directory (ex: /home/user/coco_dataset/val2017)>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/object-detection:tf-latest-rfcn-int8-inference \
  /bin/bash quickstart/int8_inference.sh
```
To get accuracy metrics:
```
DATASET_DIR=<path to the COCO validation TF record file (ex: /home/user/coco_output/coco_val.record)>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/object-detection:tf-latest-rfcn-int8-inference \
  /bin/bash quickstart/int8_accuracy.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

