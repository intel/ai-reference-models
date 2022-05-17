<!--- 0. Title -->
# PyTorch ResNet50 training

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 training using
Intel-optimized PyTorch.

## Bare Metal

### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison, Torch-CCL and Jemalloc.

### Model Specific Setup

* Set Jemalloc Preload for better performance

The jemalloc should be built from the [General setup](#general-setup) section.

```bash
    export LD_PRELOAD="path/lib/libjemalloc.so":$LD_PRELOAD
    export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
```

* Set IOMP preload for better performance

IOMP should be installed in your conda env from the [General setup](#general-setup) section.

```bash
    export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
```

* Set ENV to use AMX if you are using SPR

```bash
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```

* Set ENV to use multi-node distributed training (no need for single-node multi-sockets)

  In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes.
```bash
    export NNODES=#your_node_number
    export HOSTFILE=your_ip_list_file #one ip per line
```

## Datasets

### ImageNet

Download and extract the ImageNet2012 training and validation dataset from
[http://www.image-net.org/](http://www.image-net.org/),
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

A after running the data prep script, your folder structure should look something like this:

```txt
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

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training.sh` | Trains using one node for one epoch for the specified precision (fp32, avx-fp32, or bf16). |
| `training_dist.sh` | Distributed trains using one node for one epoch for the specified precision (fp32, avx-fp32, or bf16). |

## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory,
an output directory, precision, and the number of training epochs.

```bash
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Env vars
export DATASET_DIR=<path_to_Imagenet_Dataset>
export OUTPUT_DIR=<Where_to_save_log>
export PRECISION=<precision to run (fp32, avx-fp32, or bf16)>
export TRAINING_EPOCHS=<epoch_number(90 or other number)>

# Run the training quickstart script
cd ${MODEL_DIR}/quickstart/image_recognition/pytorch/resnet50/training/cpu
bash training.sh

# Run the distributed training quickstart script
cd ${MODEL_DIR}/quickstart/image_recognition/pytorch/resnet50/training/cpu
bash training_dist.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
