<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
<model name> <precision> <mode>. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset
and an output directory.

* Running inference to check accuracy:
```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  <docker image> \
  /bin/bash quickstart/int8_accuracy.sh
```

* Running online inference:
Set `NUM_OMP_THREADS` for tunning the hyperparameter `num_omp_threads`.

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>
NUM_OMP_THREADS=1

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env NUM_OMP_THREADS=${NUM_OMP_THREADS} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  <docker image> \
  /bin/bash quickstart/int8_online_inference.sh \
  --num-intra-threads 1 --num-inter-threads 1
```

