# BERT LARGE FP32 Inference

This document has instructions for running
[BERT](https://github.com/google-research/bert#what-is-bert) FP32 inference
using Intel-optimized TensorFlow.


## Datasets

### BERT Large Data
Download and unzip the BERT large uncased (whole word masking) model from the
    [google bert repo](https://github.com/google-research/bert#pre-trained-models).
    Then, download the Stanford Question Answering Dataset (SQuAD) dataset file `dev-v1.1.json` into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

    ```
    wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
    unzip wwm_uncased_L-24_H-1024_A-16.zip

    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
    ```
Set the `DATASET_DIR` to point to that directory when running bert large inference using the SQuAD data.

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_benchmark.sh`](fp32_benchmark.sh) | This script runs bert large fp32 inference. |
| [`fp32_profile.sh`](fp32_profile.sh) | This script runs fp32 inference in profile mode. |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | This script is runs bert large fp32 inference in accuracy mode. |

These quickstart scripts can be run the following environments:
* [Bare metal](#bare-metal)
* [Docker](#docker)

## Bare Metal

To run on bare metal, the following prerequisites must be installed in your enviornment:
* Python 3
* [intel-tensorflow==2.1.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* git

Once the above dependencies have been installed, download and untar the model
package, set environment variables, and then run a quickstart script. See the
[datasets](#datasets) and [list of quickstart scripts](#quick-start-scripts) 
for more details on the different options.

The snippet below shows how to run a quickstart script:

```
wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/bert-large-fp32-inference.tar.gz
tar -xvf bert-large-fp32-inference.tar.gz
cd bert-large-fp32-inference

DATASET_DIR=<path to the dataset being used>
OUTPUT_DIR=<directory where log files will be saved>

# Run a script for your desired usage
bash ./quickstart/<SCRIPT NAME>.sh
```

## Docker

The bert large FP32 inference model container includes the scripts and libraries
needed to run bert large FP32 inference. To run one of the quickstart scripts
using this container, you'll need to provide volume mounts for the,
dataset, and an output directory where log files will be written.

The snippet below shows how to run a quickstart script:
```
DATASET_DIR=<path to the dataset being used>
OUTPUT_DIR=<directory where log files will be saved>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  model-zoo:2.1.0-language-modeling-bert-large-fp32-inference \
  /bin/bash ./quickstart/<SCRIPT NAME>.sh
```

