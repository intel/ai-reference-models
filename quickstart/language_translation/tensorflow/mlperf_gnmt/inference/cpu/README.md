<!--- 0. Title -->
# MLPerf GNMT inference

<!-- 10. Description -->
## Description

This document has instructions for running MLPerf GNMT inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

Download and unzip the MLPerf GNMT model benchmarking data.

```
wget https://zenodo.org/record/2531868/files/gnmt_inference_data.zip
unzip gnmt_inference_data.zip
export DATASET_DIR=$(pwd)/nmt/data
```

Set the `DATASET_DIR` to point as instructed above  when running MLPerf GNMT.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`online_inference.sh`](/quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/online_inference.sh) | Runs online inference (batch_size=1). |
| [`batch_inference.sh`](/quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/batch_inference.sh) | Runs batch inference (batch_size=32). |
| [`accuracy.sh`](/quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/accuracy.sh) | Runs accuracy |

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
MLPerf GNMT inference. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory.

```
DATASET_DIR=<path to the dataset>
PRECISION=fp32
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${batch_size} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/language-translation:tf-latest-mlperf-gnmt-inference \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

