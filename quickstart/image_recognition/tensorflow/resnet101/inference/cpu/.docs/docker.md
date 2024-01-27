<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
<model name> <mode>. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory.

```
DATASET_DIR=<path to the dataset>
<<<<<<<< HEAD:quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/.docs/docker.md
PRECISION=fp32
========
PRECISION=<set the precision "int8" or "fp32">
>>>>>>>> r3.1:quickstart/image_recognition/tensorflow/resnet101/inference/cpu/.docs/docker.md
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
<<<<<<<< HEAD:quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/.docs/docker.md
  --env BATCH_SIZE=${batch_size} \
========
  --env BATCH_SIZE=${BATCH_SIZE} \
>>>>>>>> r3.1:quickstart/image_recognition/tensorflow/resnet101/inference/cpu/.docs/docker.md
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
