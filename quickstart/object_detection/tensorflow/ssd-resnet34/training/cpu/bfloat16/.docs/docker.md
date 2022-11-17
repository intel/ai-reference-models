<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
<model name> <precision> <mode>. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory where the log files and checkpoints will be written.
Use an empty output directory to prevent conflicts with checkpoint files
from previous runs. To run more than one process, set the `MPI_NUM_PROCESSES` environment
variable in the container. Depending on which quickstart script is being
run, other volume mounts or environment variables may be required.

When using the [`bfloat16_training_demo.sh`](bfloat16_training_demo.sh)
quickstart script, the `TRAIN_STEPS` (defaults to 100) environment variable
can be set in addition to the `DATASET_DIR` and `OUTPUT_DIR`. The
`MPI_NUM_PROCESSES` will default to 1 if it is not set.
```
export DATASET_DIR=<path to the COCO training data>
export OUTPUT_DIR=<directory where the log and checkpoint file will be written>
export TRAIN_STEPS=<optional, defaults to 100>
export MPI_NUM_PROCESSES=<optional, defaults to 1>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env TRAIN_STEPS=${TRAIN_STEPS} \
  --env MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  <docker image> \
  /bin/bash quickstart/bfloat16_training_demo.sh
```

To run the [`bfloat16_training.sh`](bfloat16_training.sh) quickstart script,
download the backbone model using the commands below. This directory where
the backbone model files are saved to is the `BACKBONE_MODEL_DIR` which will
get mounted in the container and set as an environment variable, just like
the `DATASET_DIR` and `OUTPUT_DIR`. The `MPI_NUM_PROCESSES` will default
to 4 if it is not set.
```
export BACKBONE_MODEL_DIR="$(pwd)/backbone_model"
mkdir -p $BACKBONE_MODEL_DIR
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/checkpoint
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.data-00000-of-00001
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.index
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.meta

export DATASET_DIR=<path to the COCO training data>
export OUTPUT_DIR=<directory where the log file and checkpoints will be written>
export MPI_NUM_PROCESSES=<optional, defaults to 4>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env BACKBONE_MODEL_DIR=${BACKBONE_MODEL_DIR} \
  --env MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${BACKBONE_MODEL_DIR}:${BACKBONE_MODEL_DIR} \
  --privileged --init -it \
  <docker image> \
  /bin/bash quickstart/bfloat16_training.sh
```

To run the [`bfloat16_training_accuracy.sh`](bfloat16_training_accuracy.sh)
quickstart script, set the `CHECKPOINT_DIR` to the directory where your
checkpoint files are located. The `CHECKPOINT_DIR` needs to get mounted in
the container and set as an environment variable, just like the `DATASET_DIR`
and `OUTPUT_DIR`. Note that when testing accuracy, the `DATASET_DIR` points
to the COCO validation dataset, instead of the training dataset. The
`MPI_NUM_PROCESSES` will default to 1 if it is not set.
```
export DATASET_DIR=<path to the COCO validation data>
export OUTPUT_DIR=<directory where the log file will be written>
export CHECKPOINT_DIR=<directory where your checkpoint files are located>
export MPI_NUM_PROCESSES=<optional, defaults to 1>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  --privileged --init -it \
  <docker image> \
  /bin/bash quickstart/bfloat16_training_accuracy.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.
