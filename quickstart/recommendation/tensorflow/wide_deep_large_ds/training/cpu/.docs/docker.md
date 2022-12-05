<!-- 60. Docker -->
## Docker

The model container used in the example below includes the scripts and
libraries needed to run <model name> <mode>. To run one of the
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
  <docker image> \
  /bin/bash quickstart/<script name>.sh
```

The script will write a log file and the saved model to the `OUTPUT_DIR`
and checkpoints will be written to the `CHECKPOINT_DIR`.

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.
