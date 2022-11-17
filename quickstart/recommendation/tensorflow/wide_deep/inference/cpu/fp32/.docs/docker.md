<!-- 60. Docker -->
### Docker

When running in docker, the <model name> <precision> <mode> container includes the model package and TensorFlow model source repo,
which is needed to run inference. To run the quickstart scripts, you'll need to provide volume mounts for the dataset and
an output directory where log files will be written.

```
DATASET_DIR=<path to the Wide & Deep dataset directory>
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
