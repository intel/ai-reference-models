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
  model-zoo:intel-python-dlrm-bf16-training \
  /bin/bash quickstart/recommendation/pytorch/dlrm/training/bf16/train_single_node.sh
```
