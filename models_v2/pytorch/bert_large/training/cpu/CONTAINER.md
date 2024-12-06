# PyTorch BERT Large training

## Description 
This document has instructions for running BERT-Large training using Intel Extension for PyTorch. 

## Pull Command

```bash
docker pull intel/language-modeling:pytorch-cpu-bert-large-training
```

> [!NOTE]
> The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets
Follow instructions to [download and preprocess](./README.md#download-the-preprocessed-text-dataset)  the text dataset and set the `DATASET_DIR` to point to the pre-processed dataset.

# BERT Config File
BERT Training happens in two stages. Download the BERT Config file from [here](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) and export `BERT_MODEL_CONFIG` variable to point to this file path. 

# Checkpoint Directory
The checkpoint directory is created as a result of Phase 1 Training. Please set the `PRETRAINED_MODEL` to point to the pre-trained model path and volume mount it for Phase 2 training. 

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
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRECISION=<specify the precision to run>
export BERT_MODEL_CONFIG=<path to bert configuration file>
export PRETRAINED_MODEL=<path to checkpoint to directory>
export TRAINING_PHASE=<set either 1 or 2>
export DNNL_MAX_CPU_ISA=<provide AVX512_CORE_AMX_FP16 for fp16 precision>
export TRAIN_SCRIPT=/workspace/pytorch-bert-large-training/run_pretrain_mlperf.py 
export DDP=false
export TORCH_INDUCTOR=0

DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/language-modeling:pytorch-cpu-bert-large-training

docker run \
  --cap-add SYS_NICE \
  --shm-size 16G \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env TRAIN_SCRIPT=${TRAIN_SCRIPT} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env TRAINING_PHASE=${TRAINING_PHASE} \
  --env DDP=${DDP} \
  --env TORCH_INDUCTOR=${TORCH_INDUCTOR} \
  --env BERT_MODEL_CONFIG=${BERT_MODEL_CONFIG} \
  --env PRETRAINED_MODEL=${PRETRAINED_MODEL} \
  --env DNNL_MAX_CPU_ISA=${DNNL_MAX_CPU_ISA} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${BERT_MODEL_CONFIG}:${BERT_MODEL_CONFIG} \
  --volume ${PRETRAINED_MODEL}:${PRETRAINED_MODEL} \
  ${DOCKER_RUN_ENVS} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash run_model.sh
```

> [!NOTE]
> The workload container was validated on a single node(`DDP=false`) with `TORCH_INDUCTOR=0`.

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/language-modeling)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/models_v2/pytorch/bert_large/training/cpu/CONTAINER.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pytorch)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
