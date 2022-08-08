<!-- 60. Docker -->
## Docker

When running in docker, the <model name> <precision> <mode> container includes the
libraries and the model package, which are needed to run <model name> <precision>
<mode>. To run the quickstart scripts, you'll need to provide volume mounts for the
[COCO validation dataset](/datasets/coco/README.md) TF Record file and an output directory
where log files will be written. Omit the `DATASET_DIR` when running the multi-instance
quickstart scripts, since synthetic data will be used.

```
DATASET_DIR=<path to the dataset (for accuracy testing only)>
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
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
