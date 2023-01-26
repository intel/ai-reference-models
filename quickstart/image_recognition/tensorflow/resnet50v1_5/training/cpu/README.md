<!--- 0. Title -->
# ResNet50 v1.5 training

<!-- 10. Description -->

This document has instructions for running ResNet50 v1.5 training
using Intel-optimized TensorFlow.


<!--- 30. Datasets -->
## Datasets

Note that the ImageNet dataset is used in these ResNet50 v1.5 examples.
Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to this directory when running ResNet50 v1.5.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`training_demo.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/training_demo.sh) | Executes a short run using small batch sizes and a limited number of steps to demonstrate the training flow |
| [`training_1_epoch.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/training_1_epoch.sh) | Executes a test run that trains the model for 1 epoch and saves checkpoint files to an output directory. |
| [`training_full.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/training_full.sh) | Trains the model using the full dataset and runs until convergence (90 epochs) and saves checkpoint files to an output directory. Note that this will take a considerable amount of time. |
| [`multi_instance_training_demo.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/multi_instance_training_demo.sh) | Uses mpirun to execute 2 processes with 1 process per socket with a batch size of 256 for 50 steps. |
| [`multi_instance_training.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/multi_instance_training.sh) | Uses mpirun to execute 1 processes with 1 process per socket with a batch size of 1024 for the specified precision (fp32 or bfloat16). Checkpoint files and logs for each instance are saved to the output directory.|

## Docker

The model container `<docker image>` includes the scripts
and libraries needed to run <model name> <mode>. To run one of the model
inference quickstart scripts using this container, you'll need to provide volume mounts for
the ImageNet dataset (if a real dataset is being used) and an output directory where
checkpoint files will be written.

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
PRECISION=<set the precision to "fp32" or "bfloat16">
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  <docker image> \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!-- 61. Advanced Options -->

See the [Advanced Options for Model Packages and Containers](/quickstart/common/tensorflow/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

