# Wide ResNet-101-2 Inference

## Description

This document has instructions for running Wide ResNet-101-2 inference using
Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison and Jemalloc.

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

## Datasets

### ImageNet

The [ImageNet](http://www.image-net.org/) validation dataset is used to run Wide ResNet-101-2
accuracy tests.

Download and extract the ImageNet2012 dataset from [http://www.image-net.org/](http://www.image-net.org/),
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

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
The folder that contains the `val` directory should be set as the
`DATASET_DIR` (for example: `export DATASET_DIR=/home/<user>/imagenet`).

## Quick Start Scripts

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash batch_inference_baremetal.sh fp32 | bash online_inference_baremetal.sh fp32 | bash accuracy_baremetal.sh fp32 |
| BF16        | bash batch_inference_baremetal.sh bf16 | bash online_inference_baremetal.sh bf16 | bash accuracy_baremetal.sh bf16 |

## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory
and an output directory.

```
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Env vars
export DATASET_DIR=<path_to_Imagenet_Dataset>
export OUTPUT_DIR=<Where_to_save_log>

# Run a quickstart script (for example, FP32 batch inference)
cd ${MODEL_DIR}/quickstart/image_recognition/pytorch/wide_resnet101_2/inference/cpu
bash batch_inference_baremetal.sh fp32
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
