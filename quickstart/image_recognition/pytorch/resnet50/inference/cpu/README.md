<!--- 0. Title -->
# PyTorch ResNet50 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 inference using
Intel-optimized PyTorch.

## Bare Metal

### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison and Jemalloc.

### Model Specific Setup

* [Datasets](#dataset)

```bash
    export DATASET_DIR=#Where_to_save_Dataset
```

* Setup the Output dir to store the log

```bash
    export OUTPUT_DIR=$Where_to_save_log
```

* Setup runnning precision

```bash
    export PRECISION=$Data_type(fp32, int8, avx-int8, or bf16)
```

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

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance with synthetic data for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |
| `inference_throughput.sh` | Runs multi instance batch inference using 1 instance per socket with synthetic data for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |

> Note: The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets

### ImageNet

The [ImageNet](http://www.image-net.org/) validation dataset is used to run ResNet50
accuracy tests.

Download and extract the ImageNet2012 dataset from [http://www.image-net.org/](http://www.image-net.org/),
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

A after running the data prep script, your folder structure should look something like this:

```txt
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

**Note:** If you don't run `accuracy.sh`, you don't need to set the environment variable `DATASET_DIR`. Performance is measured with dummy data.

## Docker

Make sure, you have all the requirements pre-setup in your Container as the [Bare Metal](#bare-metal) Setup section.

### Download dataset

Refer to the corresponding Bare Mental Section to download the dataset.

### Running CMD

```bash
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
  intel/image_recognition:pytorch-latest-resnet50-inference \
  /bin/bash quickstart/<script name>.sh <data_type>
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
