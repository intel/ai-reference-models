# BERT Large Inference

## Description

This document has instructions for running BERT Large Inference with FP16 precision using Intel(R) Extension for PyTorch on Intel Max Series GPU. 

## Datasets

### SQuAD dataset

Download the [SQuAD 1.0 dataset](https://github.com/huggingface/transformers/tree/v4.0.0/examples/question-answering#fine-tuning-bert-on-squad10).
Set the `DATASET_DIR` to point to the directory where the files are located before running the BERT quickstart scripts. Your dataset directory should look something
like this:
```
<DATASET_DIR>/
├── dev-v1.1.json
├── evaluate-v1.1.py
└── train-v1.1.json
```
The setup assumes the dataset is downloaded to the current directory. 

## Pre-trained Model

Download the `config.json` and fine tuned model from huggingface and set the `BERT_WEIGHT` environment variable to point to the directory that has both files:

```
mkdir bert_squad_model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
BERT_WEIGHT=$(pwd)/bert_squad_model
```

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `fp16_inference_plain_format.sh` | Runs BERT Large inference (plain format) for fp16 precision |

Requirements:
* Host machine has Intel(R) Data Center Max Series GPU
* Follow instructions to install GPU-compatible driver [602](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-dc.html#step-1-add-package-repository)
* Docker

## Docker pull Command

```
docker pull intel/language-modeling:pytorch-max-gpu-bert-large-inference
```

The BERT Large inference container includes scripts,models,libraries needed to run fp16 inference To run the `fp16_inference_plain_format.sh` quick start script follow the instructions below.

```
export DATASET_DIR=<path to dataset>
export OUTPUT_DIR=<path to output log files>
export BERT_WEIGHT=$(pwd)/bert_squad_model

DOCKER_ARGS=${DOCKER_ARGS:---rm -it}
IMAGE_NAME=intel/language-modeling:pytorch-max-gpu-bert-large-inference

SCRIPT=quickstart/fp16_inference_plain_format.sh
Tile=2

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env DATASET_DIR=${DATASET_DIR} \
  --env BERT_WEIGHT=${BERT_WEIGHT} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env Tile=${Tile} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${BERT_WEIGHT}:${BERT_WEIGHT} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
  ```
