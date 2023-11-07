# PyTorch Mask RCNN inference

## Description 
This document has instructions for running MaskRCNN inference using Intel Extension for PyTorch. 

## Pull Command

Docker image based on CentOS Stream8
```
docker pull intel/object-detection:pytorch-cpu-centos-maskrcnn-inference
```
Docker image based on Ubuntu 22.04
```
docker pull intel/object-detection:pytorch-cpu-ubuntu-maskrcnn-inference
```

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, avx-fp32, bf16, or bf32) and mode (imperative or jit). |
| `inference_throughput.sh` | Runs multi instance batch inference using 24 cores per instance for the specified precision (fp32, avx-fp32, bf16, or bf32) and mode (imperative or jit). |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, avx-fp32, bf16, or bf32) and mode (imperative or jit). |

**Note:** The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

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
Download the pretrained model and set the CHECKPOINT_DIR environment variable to point to the directory where the weights file is downloaded.
```
curl -O https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth
export CHECKPOINT_DIR=$(pwd)
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

To run MaskRCNN inference, set environment variables to specify the dataset directory, precision and mode to run, and an output directory.

```bash
# Set the required environment vars
export OS=<provide either centos or ubuntu>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export SCRIPT=quickstart/<specify the script to run>
export PRECISION=<specify the precision to run>
export MODE=<specify one of jit or imperative>
export CHECKPOINT_DIR=<path to the downloaded pre-trained model>

IMAGE_NAME=intel/object-detection:pytorch-cpu-${OS}-maskrcnn-inference
WORKDIR=/workspace/pytorch-maskrcnn-inference
DOCKER_ARGS="--privileged --init -it"

docker run --rm \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --shm-size 8G \
  -w ${WORKDIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT $PRECISION $MODE
```
## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/object-detection)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/quickstart/object_detection/pytorch/maskrcnn/inference/cpu/EMR_DEVCATALOG.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/docker/pyt-cpu)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
