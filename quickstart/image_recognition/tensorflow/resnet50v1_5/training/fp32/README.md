<!--- 0. Title -->
# ResNet50 v1.5 FP32 training

<!-- 10. Description -->

This document has instructions for running ResNet50 v1.5 FP32 training
using Intel-optimized TensorFlow.

Note that the ImageNet dataset is used in these ResNet50 v1.5 examples.
Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

<!--- 20. Download link -->
## Download link

[resnet50v1-5-fp32-training.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/resnet50v1-5-fp32-training.tar.gz)

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_training_demo.sh`](fp32_training_demo.sh) | Executes a short run using small batch sizes and a limited number of steps to demonstrate the training flow |
| [`fp32_training_1_epoch.sh`](fp32_training_1_epoch.sh) | Executes a test run that trains the model for 1 epoch and saves checkpoint files to an output directory. |
| [`fp32_training_full.sh`](fp32_training_full.sh) | Trains the model using the full dataset and runs until convergence (90 epochs) and saves checkpoint files to an output directory. Note that this will take a considerable amount of time. |

These quick start scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)


<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your enviornment:
* Python 3
* [intel-tensorflow==2.3.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package and then run a [quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where checkpoint and log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/resnet50v1-5-fp32-training.tar.gz
tar -xvf resnet50v1-5-fp32-training.tar.gz
cd resnet50v1-5-fp32-training

quickstart/<script name>.sh
```

<!-- 60. Docker -->
## Docker

The ResNet50 v1.5 FP32 training model container includes the scripts
and libraries needed to run ResNet50 v1.5 FP32 training. To run one of the model
training quickstart scripts using this container, you'll need to provide volume mounts for
the ImageNet dataset and an output directory where checkpoint files will be written.

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where checkpoint and log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/image-recognition:tf-2.3.0-imz-2.2.0-resnet50v1-5-fp32-training \
  /bin/bash quickstart/<script name>.sh
```


<!-- 61. Advanced Options -->

See the [Advanced Options for Model Packages and Containers](/quickstart/common/tensorflow/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

