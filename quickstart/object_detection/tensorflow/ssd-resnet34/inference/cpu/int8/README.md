<!--- 0. Title -->
# SSD-ResNet34 Int8 inference

<!-- 10. Description -->
## Description

This document has instructions for running
[SSD-ResNet34](https://arxiv.org/pdf/1512.02325.pdf) Int8 inference
using Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[ssd-resnet34-int8-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/ssd-resnet34-int8-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

SSD-ResNet34 uses the [COCO dataset](https://cocodataset.org) for accuracy
testing.

Download and preprocess the COCO validation images using the
[instructions here](/datasets/coco). After the script to convert the raw
images to the TF records file completes, rename the tf_records file:
```
mv ${OUTPUT_DIR}/coco_val.record ${OUTPUT_DIR}/validation-00000-of-00001
```

Set the `DATASET_DIR` to the folder that has the `validation-00000-of-00001`
file when running the accuracy test. Note that the inference performance
test uses synthetic dataset.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [int8_accuracy.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/int8/int8_accuracy.sh) | Tests accuracy using the COCO dataset in the TF Records format with an input size of 300x300. |
| [int8_inference.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/int8/int8_inference.sh) | Run inference using synthetic data with an input size of 300x300 and outputs performance metrics. |
| [int8_accuracy_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/int8/int8_accuracy_1200.sh) | Tests accuracy using the COCO dataset in the TF Records format with an input size of 1200x1200. |
| [multi_instance_batch_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/int8/multi_instance_batch_inference_1200.sh) | Uses numactl to run inference (batch_size=1) with an input size of 1200x1200 and one instance per socket. Waits for all instances to complete, then prints a summarized throughput value. |
| [multi_instance_online_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/int8/multi_instance_online_inference_1200.sh) | Uses numactl to run inference (batch_size=1) with an input size of 1200x1200 and four cores per instance. Waits for all instances to complete, then prints a summarized throughput value. |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* git
* libgl1-mesa-glx
* libglib2.0-0
* numpy>=1.17.4
* Cython
* contextlib2
* pillow>=8.1.2
* lxml
* jupyter
* matplotlib
* pycocotools
* horovod==0.20.0
* tensorflow-addons==0.11.0
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
git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
cd ..
```

Download the SSD-ResNet34 pretrained model for either the 300x300 or 1200x1200
input size, depending on which [quickstart script](#quick-start-scripts) you are
going to run. Set the `PRETRAINED_MODEL` environment variable for the path to the
pretrained model that you'll be using.
```
# ssd-resnet34 300x300
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_int8_bs1_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/ssd_resnet34_int8_bs1_pretrained_model.pb

# ssd-resnet34 1200x1200
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_int8_1200x1200_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/ssd_resnet34_int8_1200x1200_pretrained_model.pb
```

Set an environment variable for the path to an `OUTPUT_DIR`
where log files will be written. If the accuracy test is being run, then
also set the `DATASET_DIR` to point to the folder where the COCO dataset
`validation-00000-of-00001` file is located. Once the environment
variables are setup, then run a [quickstart script](#quick-start-scripts).

To run inference using synthetic data:
```
export OUTPUT_DIR=<directory where log files will be written>

./quickstart/int8_inference.sh
```

To test accuracy using the COCO dataset:
```
export DATASET_DIR=<path to the coco directory>
export OUTPUT_DIR=<directory where log files will be written>

./quickstart/int8_accuracy.sh
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
  intel/object-detection:tf-latest-ssd-resnet34-int8-inference \
  /bin/bash quickstart/<script name>.sh
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
  intel/object-detection:tf-latest-ssd-resnet34-int8-inference \
  /bin/bash quickstart/int8_accuracy.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

