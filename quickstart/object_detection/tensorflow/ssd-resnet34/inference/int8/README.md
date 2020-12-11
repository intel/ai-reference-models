<!--- 0. Title -->
# SSD-ResNet34 Int8 inference

<!-- 10. Description -->
## Description

This document has instructions for running
[SSD-ResNet34](https://arxiv.org/pdf/1512.02325.pdf) Int8 inference
using Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[ssd-resnet34-int8-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/ssd-resnet34-int8-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

SSD-ResNet34 uses the [COCO dataset](https://cocodataset.org) for accuracy
testing.

Download and preprocess the COCO validation images using the
[instructions here](/datasets/coco). After the script to convert the raw
images to the TF records file completes, rename the tf_records file:
```
$ mv ${OUTPUT_DIR}/coco_val.record ${OUTPUT_DIR}/validation-00000-of-00001
```

Set the `DATASET_DIR` to the folder that has the `validation-00000-of-00001`
file when running the accuracy test. Note that the inference performance
test uses synthetic dataset.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [int8_accuracy.sh](int8_accuracy.sh) | Tests accuracy using the COCO dataset in the TF Records format. |
| [int8_inference.sh](int8_inference.sh) | Run inference using synthetic data and outputs performance metrics. |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow==2.3.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* git
* libgl1-mesa-glx
* libglib2.0-0
* numpy==1.17.4
* Cython
* contextlib2
* pillow>=7.1.0
* lxml
* jupyter
* matplotlib
* pycocotools
* horovod==0.20.0
* tensorflow-addons==0.8.1
* opencv-python

In addition to the libraries above, SSD-ResNet34 uses the
[TensorFlow models](https://github.com/tensorflow/models) and
[TensorFlow benchmarks](https://github.com/tensorflow/benchmarks)
repositories. Clone the repositories using the commit ids specified
below and set the `TF_MODELS_DIR` to point to the folder where the models
repository was cloned:
```
# Clone the TensorFlow models repo
git clone https://github.com/tensorflow/models.git tf_models
cd tf_models
git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
export TF_MODELS_DIR=$(pwd)
cd ..

# Clone the TensorFlow benchmarks repo
git clone --single-branch https://github.com/tensorflow/benchmarks.git ssd-resnet-benchmarks
cd ssd-resnet-benchmarks
$ git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
$ cd ..
```

After installing the prerequisites and cloning the required repositories,
download and untar the model package. The model package includes the
SSD-ResNet34 Int8 pretrained model and the scripts needed to run
inference.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/ssd-resnet34-int8-inference.tar.gz
tar -xzf ssd-resnet34-int8-inference.tar.gz
cd ssd-resnet34-int8-inference
```

Set an environment variable for the path to an `OUTPUT_DIR`
where log files will be written. If the accuracy test is being run, then
also set the `DATASET_DIR` to point to the folder where the COCO dataset
`validation-00000-of-00001` file is located. Once the environment
variables are setup, then run a [quickstart script](#quick-start-scripts).

To run inference using synthetic data:
```
export OUTPUT_DIR=<directory where log files will be written>

quickstart/int8_inference.sh
```

To test accuracy using the COCO dataset:
```
export DATASET_DIR=<path to the coco directory>
export OUTPUT_DIR=<directory where log files will be written>

quickstart/int8_accuracy.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the pretrained model, scripts and libraries
needed to run  SSD-ResNet34 Int8 inference. To run one of the
quickstart scripts using this container, you'll need to provide a volume
mount for an output directory where logs will be written. If you are
testing accuracy, then the directory where the coco dataset
`validation-00000-of-00001` file located will also need to be mounted.

To run inference using synthetic data:
```
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/object-detection:tf-2.3.0-imz-2.2.0-ssd-resnet34-int8-inference \
  /bin/bash quickstart/int8_inference.sh
```

To test accuracy using the COCO dataset:
```
DATASET_DIR=<path to the COCO directory>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/object-detection:tf-2.3.0-imz-2.2.0-ssd-resnet34-int8-inference \
  /bin/bash quickstart/int8_accuracy.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

