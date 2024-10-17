# Running DLRMv2 Inference using Intel® Extension for PyTorch*

## Description 
This document provides instructions for running DLRMv2 inference using Intel® Extension for PyTorch on Intel® Xeon® Scalable Processors. 

## Pull Command

```bash
docker pull intel/recommendation:pytorch-cpu-dlrmv2-inference
```

* Set ENV for fp16 to leverage AMX if you are using a supported platform.

```bash
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```
* Set ENV for int8/bf32 to leverage VNNI if you are using a supported platform.
```bash
export DNNL_MAX_CPU_ISA=AVX2_VNNI_2
```

## Datasets and Pre-Trained Model

Refer to instructions [here](README.md#datasets) and [here](README.md#pre-trained-checkpoint) to download datasets and pre-trained model respectively. These two inputs are required only when testing accuracy and not required for performance calculations. 

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
To run DLRMv2 inference, set environment variables to specify the precision, dataset directory,pre-trained models and an output directory. For performance inference, `DATASET_DIR` and `WEIGHT_DIR` are optional as dummy data will be generated and model will be downloaded and used for inference. 

```bash
##Optional
export TORCH_INDUCTOR=<provide either 0 or 1, otherwise (default:0)>
export BATCH_SIZE=<provide batch size, otherwise (default:256)>
## Required
export OUTPUT_DIR=<path to output directory>
export DATASET_DIR=<path to dataset directory only required for accuracy test>
export WEIGHT_DIR=<path to pre-trained model directory only required for accuracy test>
export DNNL_MAX_CPU_ISA=<provide either AVX512_CORE_AMX_FP16 for fp16 or AVX2_VNNI_2 for int8/bf32 if supported by platform>
export PRECISION=<provide either fp32, int8, bf16 , bf32 or fp16>

DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/recommendation:pytorch-cpu-dlrmv2-inference

docker run \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env WEIGHT_DIR=${WEIGHT_DIR} \
  --env TORCH_INDUCTOR=${TORCH_INDUCTOR} \
  --env DNNL_MAX_CPU_ISA=${DNNL_MAX_CPU_ISA} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${WEIGHT_DIR}:${WEIGHT_DIR} \
  ${DOCKER_RUN_ENVS} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash run_model.sh
  ```

> [!NOTE]
> The container has been validated with`TORCH_INDUCTOR=0`.

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/recommendation)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/models_v2/pytorch/torchrec_dlrm/inference/cpu/CONTAINER.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pyt-cpu)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
