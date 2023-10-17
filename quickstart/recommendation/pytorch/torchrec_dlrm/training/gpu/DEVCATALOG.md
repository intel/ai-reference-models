# PyTorch DLRM Training

## Description 
This document has instructions for running DLRM training with BFloat16 precision on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch.

## Datasets

The dataset required to train the model is the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf). Please refer to the [link](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm#create-the-synthetic-multi-hot-dataset) to follow steps 1-3. After dataset pre-processing, set the `DATASET_DIR` environment variable to point to the dataset directory. Please note that the pre-processing step requires 700GB of RAM and takes 1-2 days to run.
## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
|`multi_card_distributed_train.sh` | Runs DLRM distributed training on single-node x4 OAM Modules |

Requirements:
* Host machine has Intel(R) Data Center Max Series 1550 x4 OAM GPU
* Follow instructions to install GPU-compatible driver [647](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html)
* Docker

## Docker pull Command

```bash
docker pull intel/recommendation:pytorch-max-gpu-dlrm-training
```
The DLRM training container includes scripts,models,libraries needed to run bf16 training. To run the `multi_card_distributed_train.sh` quickstart script follow the instructions below. 

```bash
export GLOBAL_BATCH_SIZE=<provide suitable batch size. Default is 65536>
export TOTAL_TRAINING_SAMPLES=<provide suitable number. Default is 4195197692>
export PRECISION=<provide suitable precision. Default is bf16>
export DATASET_DIR=<provide path to the Terabyte Dataset directory>
export NUM_OAM=<provide 4 for number of OAM Modules supported by the platform>
export OUTPUT_DIR=<path to output directory to view logs> 

DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/recommendation:pytorch-max-gpu-dlrm-training
SCRIPT=quickstart/multi_card_distributed_train.sh

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env NUM_OAM=${NUM_OAM} \
  --env PRECISION=${PRECISION} \
  --env GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} \
  --env TOTAL_TRAINING_SAMPLES=${TOTAL_TRAINING_SAMPLES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume /dev/dri:/dev/dri \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
```
