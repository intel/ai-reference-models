# Running GPT-J Inference using Intel® Extension for PyTorch*

## Description 
This document provides instructions for running  GPT-J inference using Intel® Extension for PyTorch on Intel® Xeon® Scalable Processors. 

## Pull Command

```bash
docker pull intel/generative-ai:pytorch-cpu-gptj-inference
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
To run GPT-J inference, set environment variables to specify the precision and an output directory.

```bash
##Optional
export BATCH_SIZE=<provide batch size, otherwise (default: 1)>
export TORCH_INDUCTOR=<provide either 0 or 1, otherwise (default:0)>
##Required
export OUTPUT_DIR=<path to output directory>
export PRECISION=<provide either fp32, int8-fp32, bf16, fp16, or bf32>
export INPUT_TOKEN=<provide input token>
export OUTPUT_TOKEN=<provide output token>
export DNNL_MAX_CPU_ISA=<provide either AVX512_CORE_AMX_FP16 for fp16 or AVX2_VNNI_2 for int8/bf32 if supported by platform>
DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/generative-ai:pytorch-cpu-gptj-inference

docker run \
  --cap-add 'SYS_NICE' \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env INPUT_TOKEN=${INPUT_TOKEN} \
  --env OUTPUT_TOKEN=${OUTPUT_TOKEN} \
  --env TORCH_INDUCTOR=${TORCH_INDUCTOR} \
  --env DNNL_MAX_CPU_ISA=${DNNL_MAX_CPU_ISA} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_RUN_ENVS} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  sh -c run_model.sh
```

> [!NOTE]
> The container has been validated with `TORCH_INDUCTOR=0`, input tokens 1024 and 2016 and output tokens 128 and 32.

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/generative-ai)


[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/models_v2/pytorch/gptj/inference/cpu/CONTAINER.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pytorch)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
