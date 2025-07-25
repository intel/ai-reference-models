# Running ResNet50_v1.5 Multi-Node Training on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch*

## Description 
This document has instructions for running ResNet50 v1.5 Multi-Node training on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch.

## Requirements
| Item | Detail |
| ------ | ------- |
| Host machine  | At least 2 Nodes with [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)  |
| Drivers | GPU-compatible drivers need to be installed: Download [Driver](https://dgpu-docs.intel.com/driver/installation.html) |
| Software | Docker* |

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_model.sh` | Runs Multi-Node ResNet50 V1.5 BF16,FP32 and TF32 training |

## Datasets
Download and extract the ImageNet2012 training and validation dataset from [http://www.image-net.org/ (http://www.image-net.org/),then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

After running the data prep script and extracting the images, your folder structure
should look something like this:
```bash
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
The folder that contains the `val` and `train` directories should be set as the `DATASET_DIR` environment variable before running the quickstart script.

## Multi-Node Setup 

Refer to [instructions](https://github.com/intel/ai-containers/tree/main/preset/deep-learning/demo/pytorch-distributed#setup-ssh) provided to configure SSH keys on worker and launcher nodes. This step ensures the launcher node has password-less access to the worker node(s).

Create a `hostfile.txt` that needs to contain the hostnames/ IP addresses of the laucher and worker node(s).

Here we provide two options for running the training workloads on multi-node, docker container and singularity container.

### Running with Docker container:
```bash
docker pull intel/image-recognition:pytorch-max-gpu-resnet50v1-5-multi-node-training
```
The ResNet50 v1.5 training container includes scripts, model and libraries needed to run BF16,FP32 and TF32 training. To run the `run_model.sh` quickstart script using this container, you'll need to provide volume mounts for the ImageNet dataset. You will need to provide an output directory where log files will be written. 

```bash
#Optional
export PRECISION=<provide either BF16,FP32 or TF32, otherwise (default: BF16)>
export BATCH_SIZE=<provide batch size, otherwise (default: 256)>
export NUM_ITERATIONS=<provide number of iterations,otherwise (default: 20)>
export NUM_PROCESS=<provide number of processes,otherwise (default: 4)>
export NUM_PROCESS_PER_NODE=<provide number of processes per node,otherwise (default: 2)>
export FI_TCP_IFACE=<provide TCP interface,otherwise (default:eno0)> # Doing `ip a` can reveal the relevant interface

#Required
export DATASET_DIR=<path to ImageNet dataset>
export OUTPUT_DIR=<path to output logs directory>
export PLATFORM=Max
export MULTI_NODE=True
export MULTI_TILE=True
export MASTER_ADDR=<provide the IP address of the launcher node>
export SSH_PORT=<provide the port number used during SSH configuration>
export SSH_WORKER=<provide path to SSH Worker Config directory>
export SSH_LAUNCHER=<provide path to SSH Launcher Config directory>
export HOSTFILE=<path to hostfile.txt file>

IMAGE_NAME=intel/image-recognition:pytorch-max-gpu-resnet50v1-5-multi-node-training
SCRIPT=run_model.sh
```
On the Worker node(s), start the OpenSSH Server as follows:
```bash
docker run --rm \
  --device=/dev/dri \
  --network=host \
  --shm-size=10G \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MULTI_NODE=${MULTI_NODE} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env MASTER_ADDR=${MASTER_ADDR} \
  --env SSH_PORT=${SSH_PORT} \
  --env NUM_PROCESS=${NUM_PROCESS} \
  --env NUM_PROCESS_PER_NODE=${NUM_PROCESS_PER_NODE} \
  --env FI_TCP_IFACE=${FI_TCP_IFACE} \
  --env PLATFORM=${PLATFORM} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env PRECISION=${PRECISION} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${SSH_WORKER}:/root/.ssh \
  --volume /dev/dri:/dev/dri \
  $IMAGE_NAME \
  bash -c '/usr/sbin/sshd -p ${SSH_PORT} -f ~/.ssh/sshd_config && sleep infinity'
  ```

  On the Launcher Node, launch the workload:
  ```bash
  docker run --rm \
  --device=/dev/dri \
  --network=host \
  --shm-size=10G \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MULTI_NODE=${MULTI_NODE} \
  --env MULTI_TILE=${MULTI_TILE} \
  --env MASTER_ADDR=${MASTER_ADDR} \
  --env SSH_PORT=${SSH_PORT} \
  --env NUM_PROCESS=${NUM_PROCESS} \
  --env NUM_PROCESS_PER_NODE=${NUM_PROCESS_PER_NODE} \
  --env HOSTFILE=${HOSTFILE} \
  --env FI_TCP_IFACE=${FI_TCP_IFACE} \
  --env PLATFORM=${PLATFORM} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env NUM_ITERATIONS=${NUM_ITERATIONS} \
  --env PRECISION=${PRECISION} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${HOSTFILE}:${HOSTFILE} \
  --volume ${SSH_LAUNCHER}:/root/.ssh \
  --volume /dev/dri:/dev/dri \
  $IMAGE_NAME \
  bash -c "./run_model.sh"
  ```

### Running with Singularity container:
The ResNet50 v1.5 training container includes scripts, model and libraries needed to run BF16,FP32 and TF32 training. To run the `run_model.sh` quickstart script using this container, you'll need to provide volume mounts for the ImageNet dataset. You will need to provide an output directory where log files will be written. 

```bash
git clone https://github.com/IntelAI/models.git
cd models/models_v2/pytorch/resnet50v1_5/training/gpu
```

On the Launcher Node, set up necessary environemtal variables:
```bash
#Optional
export PRECISION=<provide either BF16,FP32 or TF32, otherwise (default: BF16)>
export BATCH_SIZE=<provide batch size, otherwise (default: 256)>
export NUM_ITERATIONS=<provide number of iterations,otherwise (default: 20)>
export NUM_PROCESS=<provide number of processes,otherwise (default: 4)>
export NUM_PROCESS_PER_NODE=<provide number of processes per node,otherwise (default: 2)>
export FI_TCP_IFACE=<provide TCP interface,otherwise (default:ib0)> # Doing `ip a` can reveal the relevant interface; default ib0 is for TCP with IP-over-IB

#Required
export CONTAINER=Singularity
export DATASET_DIR=<path to ImageNet dataset>
export OUTPUT_DIR=<path to output logs directory>
export PLATFORM=Max
export MULTI_NODE=True
export MULTI_TILE=True
export MASTER_ADDR=<provide the IP address of the launcher node>
export SSH_PORT=<provide the port number used during SSH configuration,otherwise (default: 29500)>
export HOSTFILE=<path to hostfile.txt file>

IMAGE_NAME=intel_image-recognition_pytorch-max-gpu-resnet50v1-5-training.sif
```

On the Launcher Node, launch the workload:
```bash
FI_TCP_IFACE=${FI_TCP_IFACE} singularity exec --bind ${DATASET_DIR}:${DATASET_DIR} --bind ${OUTPUT_DIR}:${OUTPUT_DIR} --bind ${HOSTFILE}:${HOSTFILE} /scratch/helpdesk/u.yq116016/images/pytorch-max-series-multi-node-multi-card-training.sif bash -c "bash run_model.sh"
```
## Documentation and Sources

[GitHub* Repository](https://github.com/IntelAI/models/tree/master/docker/max-gpu)

## Support
Support for Intel® Extension for PyTorch* is found via the [Intel® AI Tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html). Additionally, the Intel® Extension for PyTorch* team tracks both bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues). Before submitting a suggestion or bug report, please search the GitHub issues to see if your issue has already been reported.

## License Agreement

LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/IntelAI/models/tree/master/third_party) for additional details.
