<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
<model name> <mode>. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset
and an output directory.

* Running inference to check accuracy:
```
DATASET_DIR=<path to the dataset>
PRECISION=<set the precision to "int8" or "fp32">
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  <docker image> \
  /bin/bash quickstart/fp32_accuracy.sh
  
  
```

* Running online inference:
Set `NUM_OMP_THREADS` for tunning the hyperparameter `num_omp_threads`.

```
DATASET_DIR=<path to the dataset>
PRECISION=<set the precision to "int8" or "fp32">
OUTPUT_DIR=<directory where log files will be written>
NUM_OMP_THREADS=1
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_OMP_THREADS=${NUM_OMP_THREADS} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  <docker image> \
  /bin/bash quickstart/fp32_online_inference.sh \
  --num-intra-threads 1 --num-inter-threads 1
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.
