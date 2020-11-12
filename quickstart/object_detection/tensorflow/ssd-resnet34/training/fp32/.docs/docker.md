<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
<model name> <precision> <mode>. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory where the log file will be written. To run more
than one process, set the `MPI_NUM_PROCESSES` environment variable in
the container.

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log and checkpoint files will be written>
MPI_NUM_PROCESSES=<number of MPI processes (optional, defaults to 1)>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  <docker image> \
  /bin/bash quickstart/fp32_training.sh
```
