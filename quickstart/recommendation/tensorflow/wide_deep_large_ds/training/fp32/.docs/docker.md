<!-- 60. Docker -->
## Docker

The model container used in the example below includes the scripts and
libraries needed to run <model name> <precision> <mode>. To run one of the
model quickstart scripts using this container, you'll need to provide
volume mounts for the [dataset](#dataset) and an output directory where
logs, checkpoints, and the saved model will be written.
```
DATASET_DIR=<path to the dataset directory>
OUTPUT_DIR=<directory where the logs, checkpoints, and the saved model will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  <docker image> \
  /bin/bash quickstart/<script name>.sh
```

The script will write a log file, checkpoints, and the saved model to
the `OUTPUT_DIR`.
