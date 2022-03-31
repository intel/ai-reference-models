<!--- 0. Title -->
# Wide and Deep using a large dataset FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running Wide and Deep using a large dataset FP32 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[wide-deep-large-ds-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/wide-deep-large-ds-fp32-inference.tar.gz)

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
| [`fp32_online_inference.sh`](fp32_online_inference.sh) | Runs online inference (`batch_size=1`). The `NUM_OMP_THREADS` environment variable and the hyperparameters `num-intra-threads`, `num-inter-threads` can be tuned for best performance. If `NUM_OMP_THREADS` is not set, it will default to `1`. |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | Measures the model accuracy (`batch_size=1000`). |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* numactl

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/wide-deep-large-ds-fp32-inference.tar.gz
tar -xzf wide-deep-large-ds-fp32-inference.tar.gz
cd wide-deep-large-ds-fp32-inference
```

* Running inference to check accuracy:
```
./quickstart/fp32_accuracy.sh
```
* Running online inference:
Set `NUM_OMP_THREADS` for tunning the hyperparameter `num_omp_threads`.
```
NUM_OMP_THREADS=1
./quickstart/fp32_online_inference.sh \
--num-intra-threads 1 --num-inter-threads 1
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
Wide and Deep using a large dataset FP32 inference. To run one of the quickstart scripts 
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
  intel/recommendation:tf-1.15.2-wide-deep-large-ds-fp32-inference \
  /bin/bash quickstart/fp32_accuracy.sh
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
  intel/recommendation:tf-1.15.2-wide-deep-large-ds-fp32-inference \
  /bin/bash quickstart/fp32_online_inference.sh \
  --num-intra-threads 1 --num-inter-threads 1
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

