# PyTorch DLRM Inference

## Description 
This document has instructions for running DLRM Inference with FP16 precision on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch.

## Datasets

The dataset required for inference is the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf). Please refer to the [link](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm#create-the-synthetic-multi-hot-dataset) to follow steps 1-3. After dataset pre-processing, set the `DATASET_DIR` environment variable to point to the dataset directory. Please note that the pre-processing step requires 700GB of RAM and takes 1-2 days to run.

## Pre-trained Model

Download the pre-trained model as follows:
```bash
wget https://cloud.mlcommons.org/index.php/s/XzfSeLgW8FYfR3S/download -O weigths.zip
unzip weights.zip
```
After decompressing the zip file, set `PRETRAINED_MODEL` to the pre-trained model directory. 
## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
|`multi_card_distributed_inference .sh` | Runs DLRM distributed inference on single-node x4 OAM Modules |

Requirements:
* Host machine has Intel(R) Data Center Max Series 1550 x4 OAM GPU
* Follow instructions to install GPU-compatible driver [647](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html)
* Docker

## Docker pull Command

```bash
docker pull intel/recommendation:pytorch-max-gpu-dlrm-inference
```
The DLRM inference container includes scripts,models,libraries needed to run fp16 inference. To run the `multi_card_distributed_inference.sh` quickstart script follow the instructions below. 
```bash
export GLOBAL_BATCH_SIZE=<provide suitable batch size. Default is 65536>
export TOTAL_SAMPLES=<provide suitable number. Default is 4195197692>
export PRECISION=<provide suitable precision. Default is fp16>
export PRETRAINED_MODEL=<provide path to the pre-trained model directory>
export DATASET_DIR=<provide path to the Terabyte Dataset directory>
export NUM_OAM=<provide 4 for number of OAM Modules supported by the platform>
export OUTPUT_DIR=<path to output directory to view logs> 
 
DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/recommendation:pytorch-max-gpu-dlrm-inference
SCRIPT=quickstart/multi_card_distributed_inference.sh

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env NUM_OAM=${NUM_OAM} \
  --env PRECISION=${PRECISION} \
  --env PRETRAINED_MODEL=${PRETRAINED_MODEL} \
  --env GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} \
  --env TOTAL_SAMPLES=${TOTAL_SAMPLES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume /dev/dri:/dev/dri \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${PRETRAINED_MODEL}:${PRETRAINED_MODEL} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
  ```
