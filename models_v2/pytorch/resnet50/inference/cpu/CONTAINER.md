# PyTorch ResNet50 inference

## Description 
This document has instructions for running ResNet50 inference using Intel Extension for PyTorch.

## Pull Command

```
docker pull intel/image-recognition:centos-pytorch-cpu-resnet50-inference
```

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance with synthetic data for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16). |
| `inference_throughput.sh` | Runs multi instance batch inference using 1 instance per socket with synthetic data for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16). |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16). |

**Note:** The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`,except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets
The [ImageNet](http://www.image-net.org/) validation dataset is used.

Download and extract the ImageNet2012 dataset from http://www.image-net.org/, then move validation images to labeled subfolders, using [the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

A after running the data prep script, your folder structure should look something like this:

```
imagenet
└── val
    ├── ILSVRC2012_img_val.tar
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   ├── ILSVRC2012_val_00006697.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` directory should be set as the `DATASET_DIR` (for example: `export DATASET_DIR=/home/<user>/imagenet`).

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
To run ResNet inference, set environment variables to specify the dataset directory, precision and an output directory. 
```bash
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRECISION=<specify the precision>
export SCRIPT=quickstart/<specify the quickstart script>
IMAGE_NAME=intel/image-recognition:centos-pytorch-cpu-resnet50-inference
DOCKER_ARGS="--privileged --init -it"
WORKDIR=/workspace/pytorch-resnet50-inference

docker run --rm \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  ${DOCKER_RUN_ENVS} \
  --shm-size 8G \
  -w ${WORKDIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
  ```

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/image-recognition)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/quickstart/image_recognition/pytorch/resnet50/inference/cpu/DEVCATALOG.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pyt-cpu)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
