# Running ChatGLM v3 6B Inference using Intel® Extension for PyTorch*

## Description 
This document provides instructions for running ChatGLM v3 GB inference using Intel® Extension for PyTorch on Intel® Xeon® Scalable Processors. 

## Pull Command

```bash
docker pull intel/generative-ai:pytorch-cpu-chatglm-inference
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
To run ChatGLM inference, set environment variables to specify the precision and an output directory.

```bash
##Optional
export BATCH_SIZE=<provide batch size for throughput inference, otherwise (default: 1)>
##Required
export OUTPUT_DIR=<path to output directory>
export PRECISION=<For Throughput and 1024/128 token sizes provide fp32, bf16, fp16 and int8-fp32. For Realtime and 1024/128 token sizes bf16 and fp16.  For Throughput and 2016/32 token sizes provide bf16 and fp16. For Realtime and 2016/32 token sizes provide bf16 and fp16. For Accuracy fp32, bf32, bf16, fp16, int8-fp32>
export DNNL_MAX_CPU_ISA=<provide either AVX512_CORE_AMX_FP16 for fp16 or AVX2_VNNI_2 for int8/bf32 if supported by platform>
export INPUT_TOKEN=<provide input token length>
export OUTPUT_TOKEN=<provide output token length>
export TORCH_INDUCTOR=0
export TEST_MODE=<provide either REALTIME, THROUGHPUT or ACCURACY>

DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/generative-ai:pytorch-cpu-chatglm-inference

docker run \
  --cap-add SYS_NICE \
  --env TEST_MODE=${TEST_MODE} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env TORCH_INDUCTOR=${TORCH_INDUCTOR} \
  --env DNNL_MAX_CPU_ISA=${DNNL_MAX_CPU_ISA} \
  --env INPUT_TOKEN=${INPUT_TOKEN} \
  --env OUTPUT_TOKEN=${OUTPUT_TOKEN} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_RUN_ENVS} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash run_model.sh
```

> [!NOTE]
> The workload container was validated for `TORCH_INDUCTOR=0`.

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/generative-ai)


[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/models_v2/pytorch/chatglm/inference/cpu/CONTAINER.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pytorch)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
