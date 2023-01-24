<!--- 0. Title -->
# Wide and Deep Large Dataset training

<!-- 10. Description -->

This document has instructions for training [Wide and Deep](https://arxiv.org/pdf/1606.07792.pdf)
using a large dataset using Intel-optimized TensorFlow.


<!--- 30. Datasets -->
## Dataset

The Large [Kaggle Display Advertising Challenge Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
will be used for training Wide and Deep. The [data](https://www.kaggle.com/c/criteo-display-ad-challenge/data) is from
[Criteo](https://www.criteo.com) and has a field indicating if an ad was
clicked (1) or not (0), along with integer and categorical features.

Download the Large Kaggle Display Advertising Challenge Dataset from [Criteo Labs](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) in `$DATASET_DIR`.
If the evaluation/train dataset were not available in the above link, it can be downloaded as follow:
   ```
    export DATASET_DIR=<location where dataset files will be saved>
    mkdir $DATASET_DIR && cd $DATASET_DIR
    wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
    wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
   ```
The `DATASET_DIR` environment variable will be used as the dataset directory when running [quickstart scripts](#quick-start-scripts).

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`training_check_accuracy.sh`](/quickstart/recommendation/tensorflow/wide_deep_large_ds/training/cpu/training_check_accuracy.sh) | Trains the model for a specified number of steps (default is 500) and then compare the accuracy against the accuracy specified in the `TARGET_ACCURACY` env var (ex: `export TARGET_ACCURACY=0.75`). If the accuracy is not met, then script exits with error code 1. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |
| [`training.sh`](/quickstart/recommendation/tensorflow/wide_deep_large_ds/training/cpu/training.sh) | Trains the model for 10 epochs. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |
| [`training_demo.sh`](/quickstart/recommendation/tensorflow/wide_deep_large_ds/training/cpu/training_demo.sh) | A short demo run that trains the model for 100 steps. |

<!-- 60. Docker -->
## Docker

The model container used in the example below includes the scripts and
libraries needed to run Wide and Deep Large Dataset training. To run one of the
model quickstart scripts using this container, you'll need to provide
volume mounts for the [dataset](#dataset), checkpoints, and an output
directory where logs and the saved model will be written.
```
DATASET_DIR=<path to the dataset directory>
PRECISION=fp32
OUTPUT_DIR=<directory where the logs and the saved model will be written>
CHECKPOINT_DIR=<directory where checkpoint files will be read and written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=fp32 \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  --privileged --init -t \
  intel/recommendation:tf-latest-wide-deep-large-ds-training \
  /bin/bash quickstart/<script name>.sh
```

The script will write a log file and the saved model to the `OUTPUT_DIR`
and checkpoints will be written to the `CHECKPOINT_DIR`.

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!-- 61. Advanced Options -->

See the [Advanced Options for Model Packages and Containers](/quickstart/common/tensorflow/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

