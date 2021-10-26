<!--- 0. Title -->
# PyTorch ResNet50 training

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 training using
Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison and Jemalloc.

### Model Specific Setup

* [Datasets](#dataset)
```
    export DATASET_DIR=#Where_to_save_Dataset
```

* Setup the Output dir to store the log
```
    export OUTPUT_DIR=$Where_to_save_log
```
* Setup trainig epochs
```
    export TRAINING_EPOCHS=$epoch_number(90 or other number)
```
* Setup runnning precision
```
    export PRECISION=$Data_type(fp32 or bf16)
```
* Set Jemalloc Preload for better performance

The jemalloc should be built from the [General setup](#general-setup) section.
```
    export LD_PRELOAD="path/lib/libjemalloc.so":$LD_PRELOAD
    export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
```

* Set IOMP preload for better performance

IOMP should be installed in your conda env from the [General setup](#general-setup) section.
```
    export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
```

* Set ENV to use AMX if you are using SPR
```
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training_baremetal.sh` | Trains using one node for one epoch for the specified precision (fp32 or bf16). |

## Datasets

### ImageNet

Download and extract the ImageNet2012 training and validation dataset from
[http://www.image-net.org/](http://www.image-net.org/),
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

A after running the data prep script, your folder structure should look something like this:
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
The folder that contains the `val` and `train` directories should be set as the
`DATASET_DIR` (for example: `export DATASET_DIR=/home/<user>/imagenet`).

## Docker

Make sure, you have all the requirements pre-setup in your Container as the [Bare Metal](#bare-metal) Setup section.

### Download dataset

Refer to the corresponding Bare Mental Section to download the dataset.

### Running CMD
```
DATASET_DIR=$dir/imagenet
OUTPUT_DIR=$Where_to_save_the_log
docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/image_recognition:pytorch-latest-resnet50-training
```
If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)



