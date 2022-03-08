<!--- 0. Title -->
# IPEX ICX - DLRM FP32

<!-- 10. Description -->
## Description

This container includes the ipex and pytorch modules with the DLRM model for inference. The model is trained on the Terabyte Click Logs dataset. The container is optimized for Intel ICX architecture.

<!--- 20. Datasets -->
## Datasets

The dataset used in these containers is [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/).

To prepare the dataset please for the steps described below.
(1) Go to the Criteo Terabyte Dataset website(https://labs.criteo.com/2013/12/download-terabyte-click-logs/) and accept the terms of use. 
(2) Copy the data download URL in the following page, and run :
```
    mkdir <dir/to/save/dlrm_data> && cd <dir/to/save/dlrm_data>
    curl -O <download url>/day_{$(seq -s , 0 23)}.gz
    gunzip day_*.gz
```
(2) Please remember to replace  "<dir/to/save/dlrm_data>" to any path you want to download and save the dataset.

The folder that contains the "Terabyte Click Logs" dataset should be set as the
`DATASET_DIR` when running quickstart scripts 
(for example: `export DATASET_DIR=/home/<user>/terabyte_dataset`).

<!--- 30. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`ipex_inference_latency.sh`](ipex_inference_latency.sh) | Runs online inference on data and uses ipex optimization (batch_size=1). |
| [`vanilla_inference_latency.sh`](vanilla_inference_latency.sh) | Runs online inference on datat but doesn't use ipex optimization (batch_size=1). |
| [`inference_accuracy.sh`](inference_accuracy.sh) | Measures the model accuracy (batch_size=128). |

<!--- 40. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
DLRM inference in fp32 precision.

To run the accuracy test, you will need
mount a volume and set the `DATASET_DIR` environment variable to point
to the prepped [Terabyte Click Logs Dataset](#dataset). 
The accuracy script needs the model weight. To set up the `WEIGHT_PATH` 
follow the steps described below.

```
export WEIGHT_PATH=<path to weight file>
wget -O $WEIGHT_PATH https://storage.googleapis.com/intel-optimized-tensorflow/models/icx-base-a37fb5e8/terabyte_mlperf_official.pt
```
Here, <path to weight file> can be the path you want the weight file to be downloaded at
Note that you'll need the `WEIGHT_PATH`only for calculating the accuracy

```
export DATASET_DIR=<path to the dataset folder>
export WEIGHT_PATH=<path to model weights>
docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env WEIGHT_PATH=${WEIGHT_PATH} \
  --env BASH_ENV=/root/.bash_profile \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${WEIGHT_PATH}:${WEIGHT_PATH} \
  --privileged --init -t \
  intel/recommendation:pytorch-1.5.0-rc3-icx-a37fb5e8-dlrm-fp32 \
  /bin/bash quickstart/inference_accuracy.sh
```

Model weight is not needed for rest of the scripts so you can use them as below,

```
export DATASET_DIR=<path to the dataset folder>
docker run \
  --env BASH_ENV=/root/.bash_profile \
  --env DATASET_DIR=${DATASET_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --privileged --init -t \
  intel/recommendation:pytorch-1.5.0-rc3-icx-a37fb5e8-dlrm-fp32 \
  /bin/bash quickstart/<script name>.sh
```

<!--- 50. License -->
## License

[LICENSE](/LICENSE)

