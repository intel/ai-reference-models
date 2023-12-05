# PyTorch ResNet34 inference

## Description 
This document has instructions for running ResNet34 inference using Intel Extension for PyTorch. 

## Pull Command

Docker image based on CentOS Stream8
```
docker pull intel/object-detection:centos-pytorch-cpu-ssd-resnet34-inference
```

Docker image based on Ubuntu 22.04
```
docker pull intel/object-detection:ubuntu-pytorch-cpu-ssd-resnet34-inference
```

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16). |
| `inference_throughput.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16). |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16). |

**Note:** The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`

## Datasets
Download and extract the 2017 training/validation images and annotations from the [COCO dataset website](https://cocodataset.org/#download) to a `coco` folder and unzip the files. After extracting the zip files, your dataset directory structure should look something like this:
```
coco
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017
│   ├── 000000454854.jpg
│   ├── 000000137045.jpg
│   ├── 000000129582.jpg
│   └── ...
└── val2017
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    ├── 000000000632.jpg
    └── ...
```
The parent of the `annotations`, `train2017`, and `val2017` directory (in this example `coco`) is the directory that should be used when setting the `image` environment variable for YOLOv4 (for example: `export image=/home/<user>/coco/val2017/000000581781.jpg`).In addition, we should also set the `size` environment to match the size of image.(for example: `export size=416`)

## Pre-Trained Model
Download the ResNet34 Pre-trained model from the following link. Set the `CHECKPOINT_DIR` to point to the model file. 
```
dir=$(pwd)
CHECKPOINT_DIR=${PWD}
mkdir -p ${CHECKPOINT_DIR}/pretrained
cd ${CHECKPOINT_DIR}/pretrained
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=13kWgEItsoxbVKUlkQz4ntjl1IZGk6_5Z'  -O 'resnet34-ssd1200.pth'
cd $dir
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

To run ResNet34 inference, set environment variables to specify the dataset directory, precision,checkpoint directory and an output directory. 

```bash
export OS=<provide either centos or ubuntu>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export SCRIPT=quickstart/<specify the script to run>
export PRECISION=<specify the precision to run>
export CHECKPOINT_DIR=${PWD}

IMAGE_NAME=intel/object-detection:${OS}-pytorch-cpu-ssd-resnet34-inference
WORKDIR=/workspace/pytorch-ssd-resnet34-inference
DOCKER_ARGS="--privileged --init -it"

docker run --rm \
  --env DATASET_DIR=${DATASET_DIR} \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
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
[Docker* Repository](https://hub.docker.com/r/intel/object-detection)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/DEVCATALOG.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pyt-cpu)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
