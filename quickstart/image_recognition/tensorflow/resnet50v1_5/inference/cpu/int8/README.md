<!--- 0. Title -->
# ResNet50 v1.5 Int8 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 v1.5 Int8 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[resnet50v1-5-int8-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50v1-5-int8-inference.tar.gz)

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
| [`int8_online_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/int8/int8_online_inference.sh) | Runs online inference (batch_size=1). |
| [`int8_batch_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/int8/int8_batch_inference.sh) | Runs batch inference (batch_size=128). |
| [`int8_accuracy.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/int8/int8_accuracy.sh) | Measures the model accuracy (batch_size=100). |
| [`multi_instance_batch_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/int8/multi_instance_batch_inference.sh) | Uses numactl to run batch inference (batch_size=128) with one instance per socket for 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. The script waits for all instances to complete, then prints a summarized throughput value. |
| [`multi_instance_online_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/int8/multi_instance_online_inference.sh) | Uses numactl to run online inference (batch_size=1) using four cores per instance for 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. The script waits for all instances to complete, then prints a summarized throughput value. |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
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

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50v1-5-int8-inference.tar.gz
tar -xzf resnet50v1-5-int8-inference.tar.gz
cd resnet50v1-5-int8-inference

./quickstart/<script name>.sh
```

<!--- 60. Docker -->
## Docker

The model container `intel/image-recognition:tf-latest-resnet50v1-5-int8-inference` includes the scripts and libraries needed to run
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
  intel/image-recognition:tf-latest-resnet50v1-5-int8-inference \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

