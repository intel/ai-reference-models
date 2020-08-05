!<--- 60. Docker -->
## Docker

The bert large FP32 inference model container includes the scripts and libraries
needed to run bert large FP32 inference. To run one of the quickstart scripts
using this container, you'll need to provide volume mounts for the,
dataset, and an output directory where log files will be written.

The snippet below shows how to run a quickstart script:
```
DATASET_DIR=<path to the dataset being used>
OUTPUT_DIR=<directory where log files will be saved>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  model-zoo:2.1.0-language-modeling-bert-large-fp32-inference \
  /bin/bash ./quickstart/<SCRIPT NAME>.sh
```

