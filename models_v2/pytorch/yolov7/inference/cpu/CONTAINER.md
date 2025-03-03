# Running YOLOv7 Inference using Intel® Extension for PyTorch*

## Description 
This document provides instructions for running YOLOv7 inference using Intel® Extension for PyTorch on Intel® Xeon® Scalable Processors. 

## Pull Command

```bash
docker pull intel/object-detection:pytorch-cpu-yolov7-inference
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
To run YOLOv7 inference, set environment variables to specify the precision and an output directory. Refer to instructions [here](./README.md#prepare-dataset) and [here](./README.md#download-pretrained-model) to download datasets and pre-trained models.

```bash
##Optional
export BATCH_SIZE=<provide batch size for throughput inference, otherwise (default: 40)>
##Required
export OUTPUT_DIR=<path to output directory>
export PRECISION=<provide either fp32, int8, bf16, fp16, or bf32>
export DNNL_MAX_CPU_ISA=<provide either AVX512_CORE_AMX_FP16 for fp16 or AVX2_VNNI_2 for int8/bf32 if supported by platform>
export DATASET_DIR=<path to COCO dataset>
export CHECKPOINT_DIR=<path to yolov7 pre-trained model>
export TORCH_INDUCTOR=0
export TEST_MODE=<provide either REALTIME, THROUGHPUT or ACCURACY mode>

DOCKER_ARGS="--rm -it"
IMAGE_NAME=intel/object-detection:pytorch-cpu-yolov7-inference

docker run \
  --cap-add SYS_NICE \
  --env TEST_MODE=${TEST_MODE} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env TORCH_INDUCTOR=${TORCH_INDUCTOR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env DNNL_MAX_CPU_ISA=${DNNL_MAX_CPU_ISA} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  ${DOCKER_RUN_ENVS} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash run_model.sh
```

> [!NOTE]
> The workload container was validated for `TORCH_INDUCTOR=0`. 

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/object-detection)


[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/models_v2/pytorch/yolov7/inference/cpu/CONTAINER.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pytorch)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
