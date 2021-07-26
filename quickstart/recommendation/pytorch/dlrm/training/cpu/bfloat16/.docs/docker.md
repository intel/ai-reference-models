<!--- 50. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
DLRM bf16 training. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset.
```
DATA_PATH=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATA_PATH=${DATA_PATH} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATA_PATH}:${DATA_PATH} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/recommendation:pytorch-1.5.0-rc3-dlrm-bfloat16-training \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.
