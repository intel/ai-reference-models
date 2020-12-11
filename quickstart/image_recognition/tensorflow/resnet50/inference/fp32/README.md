<!--- 0. Title -->
# ResNet50 FP32 inference

<!-- 10. Description -->

This document has instructions for running ResNet50 FP32 inference using
Intel-optimized TensorFlow.

Note that the ImageNet dataset is used in these ResNet50 examples.
Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

<!--- 20. Download link -->
## Download link

[resnet50-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/resnet50-fp32-inference.tar.gz)

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](fp32_online_inference.sh) | Runs online inference (batch_size=1). |
| [`fp32_batch_inference.sh`](fp32_batch_inference.sh) | Runs batch inference (batch_size=128). |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | Measures the model accuracy (batch_size=100). |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)


<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow==2.3.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package and then run a [quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/resnet50-fp32-inference.tar.gz
tar -xzf resnet50-fp32-inference.tar.gz
cd resnet50-fp32-inference

quickstart/<script name>.sh
```


<!-- 60. Docker -->
## Docker

The model container `intel/image-recognition:tf-2.3.0-imz-2.2.0-resnet50-fp32-inference` includes the scripts
and libraries needed to run ResNet50 FP32 inference. To run one of the model
inference quickstart scripts using this container, you'll need to provide volume mounts for
the ImageNet dataset and an output directory where checkpoint files will be written.

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/image-recognition:tf-2.3.0-imz-2.2.0-resnet50-fp32-inference \
  /bin/bash quickstart/<script name>.sh
```


<!-- 61. Advanced Options -->

See the [Advanced Options for Model Packages and Containers](/quickstart/common/tensorflow/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

