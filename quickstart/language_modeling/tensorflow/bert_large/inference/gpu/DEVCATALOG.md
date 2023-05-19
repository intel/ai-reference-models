# TensorFlow BERT Large Inference

## Description
This document has instructions for running BERT Large inference with FP16 and FP32 precision using Intel® Data Center GPU Max Series. 

## Datasets 

### BERT Large Data
Download and unzip the BERT Large uncased (whole word masking) model from the [google bert repo](https://github.com/google-research/bert#pre-trained-models).
Then, download the Stanford Question Answering Dataset (SQuAD) dataset file `dev-v1.1.json` into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

```
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```
Set the `PRETRAINED_DIR` to point to that directory when running BERT Large inference using the SQuAD data.

Download the SQUAD directory and set the `SQUAD_DIR` environment variable to point where it was saved:
  ```
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
  wget https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
  ```

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `benchmark.sh` | This script runs bert large fp16 and fp32 inference. |

## Docker
Requirements:
* Host machine has Intel(R) Data Center Max Series GPU
* Follow instructions to install GPU-compatible driver [602](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-dc.html#step-1-add-package-repository)
* Docker

## Docker pull Command
```
docker pull intel/language-modeling:tf-max-gpu-bert-large-inference
```
The BERT Large Inference container includes scripts,models,libraries needed to run fp16/fp32 Inference. 

```
export PRECISION=fp16
export OUTPUT_DIR=<path to output logs>
export PRETRAINED_DIR=<path to dataset>
export SQAUD_DIR=<path to squad directory>
export Tile=2

IMAGE_NAME=intel/language-modeling:tf-max-gpu-bert-large-inference
DOCKER_ARGS="--rm -it"
SCRIPT=benchmark.sh

FROZEN_GRAPH=/workspace/tf-max-series-bert-large-inference/frozen_graph/fp32_bert_squad.pb


docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env FROZEN_GRAPH=${FROZEN_GRAPH} \
  --env PRETRAINED_DIR=${PRETRAINED_DIR} \
  --env SQUAD_DIR=${SQUAD_DIR} \
  --env Tile=${Tile} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${PRETRAINED_DIR}:${PRETRAINED_DIR} \
  --volume ${SQUAD_DIR}:${SQUAD_DIR} \
  ${DOCKER_ARGS}\
  $IMAGE_NAME \
  /bin/bash quickstart/$SCRIPT
  ```