<!--- 0. Title -->
# NCF FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running Neural Collaborative Filtering (NCF)
FP32 inference using Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[ncf-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/ncf-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

Download [movielens 1M dataset](https://grouplens.org/datasets/movielens/1m/)
```
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```
Set the `DATASET_DIR` to point to this directory when running NCF.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](fp32_online_inference.sh) | Runs online inference (batch_size=1). |
| [`fp32_batch_inference.sh`](fp32_batch_inference.sh) | Runs batch inference (batch_size=256). |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | Measures the model accuracy (batch_size=256). |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* numactl
* google-api-python-client==1.6.7
* google-cloud-bigquery==0.31.0
* kaggle==1.3.9
* numpy==1.16.3
* oauth2client==4.1.2
* pandas
* 'psutil>=5.6.7'
* py-cpuinfo==3.3.0
* typing
* TensorFlow models, clone the official `tensorflow/models` repository with  tag `v1.11`:
```
git clone https://github.com/tensorflow/models.git tf_models
cd tf_models
git checkout v1.11
```

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>
TF_MODELS_DIR=<path to the TensorFlow models directory tf_models>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/ncf-fp32-inference.tar.gz
tar -xzf ncf-fp32-inference.tar.gz
cd ncf-fp32-inference

./quickstart/<script name>.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
NCF FP32 inference. To run one of the quickstart scripts 
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
  intel/recommendation:tf-1.15.2-ncf-fp32-inference \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

