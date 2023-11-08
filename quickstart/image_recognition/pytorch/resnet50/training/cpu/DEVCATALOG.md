# PyTorch ResNet50 training

## Description 
This document has instructions for running ResNet50 training using Intel Extension for PyTorch.

## Pull Command

Docker image based on CentOS Stream8
```
docker pull intel/image-recognition:pytorch-cpu-centos-resnet50-training
```

Docker image based on Ubuntu 22.04
```
docker pull intel/image-recognition:pytorch-cpu-ubuntu-resnet50-training
```

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `training.sh` | Trains using one node for one epoch for the specified precision (fp32, avx-fp32,bf32 or bf16). |

**Note:** The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets
Download and extract the ImageNet2012 training and validation dataset from [http://www.image-net.org/](http://www.image-net.org/), then move validation images to labeled subfolders, using [the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

After running the data prep script, your folder structure should look something like this:
```
imagenet
├── train
│   ├── n02085620
│   │   ├── n02085620_10074.JPEG
│   │   ├── n02085620_10131.JPEG
│   │   ├── n02085620_10621.JPEG
│   │   └── ...
│   └── ...
└── val
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` and `train` directories should be set as the`DATASET_DIR` (for example: `export DATASET_DIR=/home/<user>/imagenet`).

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

To run ResNet50 training, set environment variables to specify the dataset directory, precision to run, and an output directory. 

```bash
export OS=<provide either centos or ubuntu>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRECISION=<specify the precision to run>
export SCRIPT=quickstart/training.sh 

IMAGE_NAME=intel/image-recognition:pytorch-cpu-${OS}-resnet50-training
WORKDIR=/workspace/pytorch-resnet50-training
DOCKER_ARGS="--privileged --init -it"
TRAINING_EPOCHS=1

docker run --rm \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env TRAINING_EPOCHS=${TRAINING_EPOCHS} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
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

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/quickstart/image_recognition/pytorch/resnet50/training/cpu/DEVCATALOG.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pyt-cpu)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
