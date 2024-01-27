# Running BERT Large inference on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch*

## Description

This document has instructions for running BERT Large Inference with FP16, BF16 and FP32 precisions using Intel(R) Extension for PyTorch on Intel Max Series GPU. 


## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html) |
| Drivers | GPU-compatible drivers need to be installed: Download [Driver](https://dgpu-docs.intel.com/driver/installation.html) |
| Software | Docker* |

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_model.sh` |  BERT Large FP16, BF16 and FP32 inference on single tile and two tiles |

## Datasets

### SQuAD dataset

Download the [SQuAD 1.0 dataset](https://github.com/huggingface/transformers/tree/v4.0.0/examples/question-answering#fine-tuning-bert-on-squad10).
Set the `DATASET_DIR` to point to the directory where the files are located before running the BERT quickstart scripts. Your dataset directory should look something
like this:
```bash
<DATASET_DIR>/
├── dev-v1.1.json
├── evaluate-v1.1.py
└── train-v1.1.json
```
The setup assumes the dataset is downloaded to the current directory. 

## Pre-trained Model

Download the `config.json` and fine tuned model from huggingface and set the `BERT_WEIGHT` environment variable to point to the directory that has both files:

```bash
mkdir squad_large_finetuned_checkpoint
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json -O squad_large_finetuned_checkpoint/config.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/pytorch_model.bin  -O squad_large_finetuned_checkpoint/pytorch_model.bin
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json -O squad_large_finetuned_checkpoint/tokenizer.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer_config.json -O squad_large_finetuned_checkpoint/tokenizer_config.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt -O squad_large_finetuned_checkpoint/vocab.txt

BERT_WEIGHT=$(pwd)/squad_large_finetuned_checkpoint
```

## Docker pull Command

```bash
docker pull intel/language-modeling:pytorch-max-gpu-bert-large-inference
```

The BERT Large inference container includes scripts, models,libraries needed to run inference To run the `fp16_inference_plain_format.sh` quick start script follow the instructions below.

```bash
#Optional
export PRECISION=<provide FP16 ,BF16 or FP32,otherwise (default:BF16)>
export BATCH_SIZE=<provide batch size,otherwise (default: 256)>
export NUM_ITERATIONS=<provide num_iterations,otherwise (default: 1)>

#Required
export MULTI_TILE=<provide True or False to enable/disable multi-tile inference>
export DATASET_DIR=<path to dataset>
export OUTPUT_DIR=<path to output log files>
export BERT_WEIGHT=$(pwd)/squad_large_finetuned_checkpoint
export PLATFORM=Max


DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/language-modeling:pytorch-max-gpu-bert-large-inference
SCRIPT=run_model.sh

docker run \
  --privileged \
  --device=/dev/dri \
  --ipc=host \
  --env DATASET_DIR=${DATASET_DIR} \
  --env BERT_WEIGHT=${BERT_WEIGHT} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env PRECISION=${PRECISION} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env PLATFORM=${PLATFORM} \
  --env MULTI_TILE=${MULTI_TILE} \
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
## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/max-gpu)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Analytics Toolkit.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.qbretz) Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
