<!--- 0. Title -->
# Wide and Deep using a large dataset inference

<!-- 10. Description -->
## Description

This document has instructions for running Wide and Deep using a large dataset inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/large_kaggle_advertising_challenge/README.md)
to download and preprocess the Large Kaggle Display Advertising Challenge Dataset.

Then, set the `DATASET_DIR` to point to this directory when running Wide and Deep using a large dataset:
```
export DATASET_DIR=/home/<user>/dataset/eval_preprocessed_eval.tfrecords
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`online_inference.sh`](/quickstart/recommendation/tensorflow/wide_deep_large_ds/inference/cpu/online_inference.sh) | Runs online inference (`batch_size=1`). The `NUM_OMP_THREADS` environment variable and the hyperparameters `num-intra-threads`, `num-inter-threads` can be tuned for best performance. If `NUM_OMP_THREADS` is not set, it will default to `1`. |
| [`accuracy.sh`](/quickstart/recommendation/tensorflow/wide_deep_large_ds/inference/cpu/accuracy.sh) | Measures the model accuracy (`batch_size=1000`). |

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
Wide and Deep using a large dataset inference. To run one of the quickstart scripts 
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
  intel/recommendation:tf-1.15.2-wide-deep-large-ds-inference \
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
  intel/recommendation:tf-1.15.2-wide-deep-large-ds-inference \
  /bin/bash quickstart/fp32_online_inference.sh \
  --num-intra-threads 1 --num-inter-threads 1
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

