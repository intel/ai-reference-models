<!--- 0. Title -->
# ResNet50 v1.5 BFloat16 training

<!-- 10. Description -->

This document has instructions for running ResNet50 v1.5 BFloat16 training
using Intel-optimized TensorFlow.


<!--- 20. Download link -->
## Download link

[resnet50v1-5-bfloat16-training.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50v1-5-bfloat16-training.tar.gz)

<!--- 30. Datasets -->
## Datasets

Note that the ImageNet dataset is used in these ResNet50 v1.5 examples.
Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to this directory when running ResNet50 v1.5.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`bfloat16_training_demo.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/bfloat16/bfloat16_training_demo.sh) | Executes a short run using small batch sizes and a limited number of steps to demonstrate the training flow |
| [`bfloat16_training_1_epoch.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/bfloat16/bfloat16_training_1_epoch.sh) | Executes a test run that trains the model for 1 epoch and saves checkpoint files to an output directory. |
| [`bfloat16_training_full.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/bfloat16/bfloat16_training_full.sh) | Trains the model using the full dataset and runs until convergence (90 epochs) and saves checkpoint files to an output directory. Note that this will take a considerable amount of time. |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package and then run a [quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where checkpoint and log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50v1-5-bfloat16-training.tar.gz
tar -xvf resnet50v1-5-bfloat16-training.tar.gz
cd resnet50v1-5-bfloat16-training

./quickstart/<script name>.sh
```

To run distributed training (one MPI process per socket) for better throughput,
set the MPI_NUM_PROCESSES var to the number of sockets to use. 
To run with multiple instances, these additional dependencies will need to be
installed in your environment:

* openmpi-bin
* openmpi-common
* openssh-client
* openssh-server
* libopenmpi-dev
* horovod==0.19.1

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where checkpoint and log files will be written>
MPI_NUM_PROCESSES=<number of sockets to use>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50v1-5-bfloat16-training.tar.gz
tar -xvf resnet50v1-5-bfloat16-training.tar.gz
cd resnet50v1-5-bfloat16-training

./quickstart/<script name>.sh
```

<!-- 60. Docker -->
## Docker

The ResNet50 v1.5 BFloat16 training model container includes the scripts
and libraries needed to run ResNet50 v1.5 BFloat16 training. To run one of the model
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
  intel/image-recognition:tf-latest-resnet50v1-5-bfloat16-training \
  /bin/bash quickstart/<script name>.sh
```

To run distributed training (one MPI process per socket) for better throughput,
set the MPI_NUM_PROCESSES var to the number of sockets to use.

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where checkpoint and log files will be written>
MPI_NUM_PROCESSES=<number of sockets to use>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES} \
  --env http_proxy=${http_proxy} --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/image-recognition:tf-latest-resnet50v1-5-bfloat16-training \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

