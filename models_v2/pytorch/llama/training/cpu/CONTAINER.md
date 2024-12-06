# Running Llama2 7B Training using Intel® Extension for PyTorch*

## Description 
This document provides instructions for running Llama2 7B training using Intel® Extension for PyTorch on Intel® Xeon® Scalable Processors. 

## Pull Command

```bash
docker pull intel/generative-ai:pytorch-cpu-llama2-training
```

* Set ENV for fp16 to leverage AMX if you are using a supported platform.

```bash
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```

* Set ENV for int8/bf32 to leverage VNNI if you are using a supported platform.
```bash
export DNNL_MAX_CPU_ISA=AVX2_VNNI_2
```

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

> [!NOTE]
> To run Llama2 7B training tests, you will need to apply for access in the pages with your huggingface account:

    - LLaMA2 7B : https://huggingface.co/meta-llama/Llama-2-7b-hf 

To run Llama2 7B training, set environment variables to specify the precision and an output directory.

Use the following instructions to download the dataset and set the environment variable `DATASET_DIR` to point to the dataset directory.

```bash
wget https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data.json -O ${DATASET_DIR}
wget https://raw.githubusercontent.com/tloen/alpaca-lora/main/templates/alpaca.json -O ${DATASET_DIR}
```

```bash
##Optional
export BATCH_SIZE=<provide batch size, otherwise (default: 32)>
export FINETUNED_MODEL=meta-llama/Llama-2-7b-hf
##Required
export OUTPUT_DIR=<path to output directory>
export PRECISION=<provide either fp32, bf16, fp16, or bf32>
export TORCH_INDUCTOR=0
export NNODES=1
export DDP=False
export DNNL_MAX_CPU_ISA=<provide either AVX512_CORE_AMX_FP16 for fp16 or AVX2_VNNI_2 for int8/bf32 if supported by platform>
export DATASET_DIR=<path to apalaca dataset>

DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/generative-ai:pytorch-cpu-llama2-training
TOKEN=<provide hugging face token">
SCRIPT="huggingface-cli login --token ${TOKEN} && ./run_model.sh"

docker run \
  --cap-add 'SYS_NICE' \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env FINETUNED_MODEL=${FINETUNED_MODEL}
  --env TORCH_INDUCTOR=${TORCH_INDUCTOR} \
  --env DNNL_MAX_CPU_ISA=${DNNL_MAX_CPU_ISA} \
  --env NNODES=${NNODES} \
  --env DDP=${DDP} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  ${DOCKER_RUN_ENVS} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  sh -c "$SCRIPT"
```

> [!NOTE]
> The container has been validated with `TORCH_INDUCTOR=0`, and on a single node(`DDP=False`).

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/generative-ai)


[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/models_v2/pytorch/llama/training/cpu/CONTAINER.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pytorch)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
