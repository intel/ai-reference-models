# PyTorch BERT Large training

## Description 
This document has instructions for running BERT-Large training using Intel Extension for PyTorch. 

## Pull Command

Docker image based on CentOS Stream8
```
docker pull intel/language-modeling:centos-pytorch-cpu-bert-large-training
```

Docker image based on Ubuntu 22.04
```
docker pull intel/language-modeling:ubuntu-pytorch-cpu-bert-large-training
```

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `run_bert_pretrain_phase1.sh` | Runs BERT large pretraining phase 1 using max_seq_len=128 for the first 90% dataset for the specified precision (fp32, avx-fp32, bf32 or bf16). The script saves the model to the `OUTPUT_DIR` in a directory called `model_save`. |
| `run_bert_pretrain_phase2.sh` | Runs BERT large pretraining phase 2 using max_seq_len=512 with the remaining 10% of the dataset for the specified precision (fp32, avx-fp32, bf32 or bf16). Use path to the `model_save` directory from phase one as the `CHECKPOINT_DIR` for phase 2. |

**Note:** The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets
Follow instructions to [download and preprocess](https://github.com/IntelAI/models/blob/v2.9.0/quickstart/language_modeling/pytorch/bert_large/training/cpu/README.md#datasets)  the text dataset and set the `DATASET_DIR` to point to the pre-processed dataset.

# BERT Config File
BERT Training happens in two stages. For stage 1, download the BERT Config file from [here](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) and export `CONFIG_FILE` variable to point to this file path. 

# Checkpoint Directory
The checkpoint directory is created as a result of Phase 1 Training. Please set the `CHECKPOINT_DIR` to point to the pre-trained model path and volume mount it for Phase 2 training. 

## Docker Run
(Optional) Export related proxy into docker environment.
```bash
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```
To run the BERT-Large training scripts, set environment variables to specify the dataset directory, precision and an output directory. 

```bash
export OS=<provide either centos or ubuntu>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export SCRIPT=quickstart/<specify the training script>
export PRECISION=<specify the precision to run>
export BERT_MODEL_CONFIG=<path to bert configuration file>
export CHECKPOINT_DIR=<path to checkpoint to Directory>

DOCKER_ARGS="--privileged --init -it"
IMAGE_NAME=intel/language-modeling:${OS}-pytorch-cpu-bert-large-training
WORKDIR=/workspace/pytorch-bert-large-training
TRAIN_SCRIPT=${WORKDIR}/models/language_modeling/pytorch/bert_large/training/run_pretrain_mlperf.py  

docker run --rm \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env TRAIN_SCRIPT=${TRAIN_SCRIPT} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env BERT_MODEL_CONFIG=${BERT_MODEL_CONFIG} \
  --env PRETRAINED_MODEL=${CHECKPOINT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${BERT_MODEL_CONFIG}:${BERT_MODEL_CONFIG} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  ${DOCKER_RUN_ENVS} \
  --shm-size 8G \
  -w ${WORKDIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT $PRECISION
```
## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/language-modeling)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/quickstart/language_modeling/pytorch/bert_large/training/cpu/DEVCATALOG.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pyt-cpu)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
