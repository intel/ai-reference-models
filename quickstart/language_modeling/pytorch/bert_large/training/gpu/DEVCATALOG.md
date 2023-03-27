# BERT Large Training 

## Description
This document has instructions for running BERT Large training with BF16 precision using Intel(R) Extension for PyTorch on Intel Max Series GPU. 

## Datasets
### Download and Extract the Dataset
Download the [MLCommons BERT Dataset](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v) and download the `results_text.tar.gz` file. Extract the file. After this step, you should have a directory called `results4` that contains 502 files with a total size of 13GB. Set the `DATASET_DIR` to point the location of this dataset. The script assumes the `DATASET_DIR` to be the current working directory. 

The above step is optional. If the `results4` folder is not present in the `DATASET_DIR` path, the quick start scripts automatically download it. 

### Generate the BERT Input Dataset 
The Training script processes the raw dataset. The processed dataset occupies about `539GB` worth of disk space. Additionally, this step can take several hours to complete to generate a folder `hdf5_seq_512`. Hence, the script provides the ability to process the data only once and this data can be volume mounted to the container for future use. Set the `PROCESSED_DATASET_DIR` to point to the location of `hdf5_seq_512`. 

The script assumes the `PROCESSED_DATASET_DIR` to be the current working directory. If the processed folder `hdf5_seq_512` does not exist in the `PROCESSED_DATASET_DIR` path, the quick start scripts process the data.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `bf16_training_plain_format.sh` | Runs BERT Large BF16 training (plain format) on two tiles |
| `ddp_bf16_training_plain_format.sh` | Runs BERT Large Distributed Data Parallel BF16 Training on two tiles | 

## Docker
Requirements:
* Host machine has Intel(R) Data Center Max Series GPU
* Follow instructions to install GPU-compatible driver [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html#ubuntu-22-04)
* Docker

## Docker pull Command
```
docker pull intel/language-modeling:pytorch-max-gpu-bert-large-training
```

The BERT Large training container includes scripts,models,libraries needed to run BF16 training. To run the `ddp_bf16_training_plain_format.sh` quick start script follow the instructions below.
```
export OUTPUT_DIR=${PWD}/logs 
export DATASET_DIR=${PWD}
export PROCESSED_DATASET_DIR=${PWD}

DOCKER_ARGS="--rm --init -it"
IMAGE_NAME=intel/language-modeling:pytorch-max-gpu-bert-large-training

VIDEO=$(getent group video | sed -E 's,^video:[^:]*:([^:]*):.*$,\1,')
RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,')
test -z "$RENDER" || RENDER_GROUP="--group-add ${RENDER}"

SCRIPT=quickstart/ddp_bf16_training_plain_format.sh
Tile=2

docker run \
  --group-add ${VIDEO} \
  ${RENDER_GROUP} \
  --device=/dev/dri \
  --shm-size=10G \
  --privileged \
  --ipc host \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PROCESSED_DATASET_DIR=${PROCESSED_DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env Tile=${Tile} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${PROCESSED_DATASET_DIR}:${PROCESSED_DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume /dev/dri:/dev/dri/ \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
  ```