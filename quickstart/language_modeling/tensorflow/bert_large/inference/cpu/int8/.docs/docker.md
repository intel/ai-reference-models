<!-- 60. Docker -->
## Docker

The <model name> <precision> <mode> model container includes the scripts,
pretrained model, and dependencies needed to run <model name> <precision>
<mode>. To run one of the quickstart scripts using this container, you'll
need to provide volume mounts for the dataset and an output directory
where log files will be written.

The snippet below shows how to run a quickstart script:
```
DATASET_DIR=<path to the SQuAD dataset>
OUTPUT_DIR=<directory where log files will be saved>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  <docker image> \
  /bin/bash ./quickstart/<script name>.sh
```
