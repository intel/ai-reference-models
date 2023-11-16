# PyTorch SSD-ResNet34 training

## Description 
This document has instructions for running SSD-ResNet34 training using Intel Extension for PyTorch. 

## Pull Command

Docker image based on CentOS Stream8
```
docker pull intel/object-detection:centos-pytorch-cpu-ssd-resnet34-training
```
Docker image based on Ubuntu 22.04
```
docker pull intel/object-detection:ubuntu-pytorch-cpu-ssd-resnet34-training
```

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `throughput.sh` | Tests the training performance for SSD-ResNet34 for the specified precision (fp32, avx-fp32, bf32 or bf16). |
| `accuracy.sh` | Tests the training accuracy for SSD-ResNet34 for the specified precision (fp32, avx-fp32, bf32 or bf16). |

**Note**: The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

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

## Pre-trained Model
Download the pre-trained model and set `CHECKPOINT_DIR`
```
CHECKPOINT_DIR=${PWD}
mkdir -p ${CHECKPOINT_DIR}/ssd; cd ${CHECKPOINT_DIR}/ssd
curl -O https://download.pytorch.org/models/resnet34-333f7ec4.pth
cd -
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

To run SSD-ResNet34, set environment variables to specify the dataset directory, precision and an output directory. 

```
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export SCRIPT=quickstart/<specify the script to run>
export PRECISION=<specify the precision to run>
export CHECKPOINT_DIR=<path to pre-trained model>

IMAGE_NAME=intel/object-detection:${OS}-pytorch-cpu-ssd-resnet34-training
DOCKER_ARGS="--privileged --init -it"
WORKDIR=/workspace/pytorch-ssd-resnet34-training

docker run --rm \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
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

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/DEVCATALOG.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pyt-cpu)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
