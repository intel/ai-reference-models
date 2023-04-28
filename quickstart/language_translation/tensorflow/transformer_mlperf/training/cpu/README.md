<!--- 0. Title -->
# Transformer Language FP32 training

<!-- 10. Description -->
## Description

This document has instructions for running Transformer Language FP32 training in mlperf
Benchmark suits using Intel-optimized TensorFlow.

Detailed information on mlperf Benchmark can be found in [mlperf/training](https://github.com/mlperf/training/tree/master/translation/tensorflow/transformer)

<!--- 30. Datasets -->
## Datasets

Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/transformer_data/README.md) to download and preprocess the WMT English-German dataset.
Set `DATASET_DIR` to point out to the location of the dataset directory.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

Transformer Language in mlperf benchmark can run with full training or
fewer training steps. During training we can control if it will do the evaluation
or not.

| Script name | Description |
|-------------|-------------|
| [`training_demo.sh`](/quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/training_demo.sh) | Runs 100 training steps. The script runs in single instance mode by default, for multi instance mode set `MPI_NUM_PROCESSES`. |
| [`training.sh`](/quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/training.sh) | Uses mpirun to execute 2 processes with 1 process per socket with a batch size of 5120 for the specified precision (fp32 or bfloat16). Logs for each instance are saved to the output directory. |

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
Transformer Language training. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory.

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
  intel/language-translation:tf-latest-transformer-mlperf-training \
  /bin/bash quickstart/<script name>
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

