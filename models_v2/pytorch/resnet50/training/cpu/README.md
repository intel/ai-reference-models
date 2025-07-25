# Resnet50 Training

This document has instructions for running ResNet50 inference using
Intel-optimized PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/KaimingHe/deep-residual-networks      |           -           |         -          |

## Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#introduction)

## Bare Metal
### General setup

[Follow](/docs/general/pytorch/BareMetalSetup.md) to install Miniforge, Pytorch, IPEX, TorchVison, Torch-CCL and Tcmalloc.

* Install dependencies
```
conda install -c conda-forge accimage
```
### Model Specific Setup

* Set Jemalloc and tcmalloc Preload for better performance

  The jemalloc should be built from the [General setup](#general-setup) section.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"path_to/tcmalloc/lib/libtcmalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```
* Set IOMP preload for better performance
```
  pip install packaging intel-openmp
  export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
```

* Set ENV to use fp16 AMX if you are using a supported platform
```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```

* Set ENV for model and dataset path, and optionally run with no network support

## Prepare Dataset

The [ImageNet](http://www.image-net.org/) validation dataset is used when
testing accuracy. The inference scripts use synthetic data, so no dataset
is needed.

Download and extract the ImageNet2012 dataset from http://www.image-net.org/,
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

The accuracy script looks for a folder named `val`, so after running the
data prep script, your folder structure should look something like this:

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
# Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/resnet50/training/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation)
5. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **DISTRIBUTED** (optional)              | `export DISTRIBUTED=True`                  |
| **DATASET_DIR**             | `export DATASET_DIR=<path to ImageNet>`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=$PWD`                               |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, avx-fp32, bf16, bf32, or fp16) |                              |
| **TRAINING_EPOCHS**     |                  `export TRAINING_EPOCHS=1` |                              |
| **LOCAL_BATCH_SIZE** (required for DISTRIBUTED)              | `    export LOCAL_BATCH_SIZE=#local batch_size(for lars optimizer convergency test, the GLOBAL_BATCH_SIZE should be 3264)`                  |
| **NNODES** (required for DISTRIBUTED)              | ` export NNODES=#your_node_number`                  |
| **HOSTFILE** (required for DISTRIBUTED)              | `export HOSTFILE=#your_ip_list_file #one ip per line`                  |
| **MASTER_ADDR** (required for DISTRIBUTED)              | `export MASTER_ADDR=#your_master_addr`                  |
| **BATCH_SIZE** (required for single-node)    |                               `export BATCH_SIZE=102`                                |
6. Run `run_model.sh`

## Output

Inference Throughput output will typically looks like (note that accuracy is measured in throughput and realtime):

```
Training throughput: 51.200 fps
```
Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 315.657
   unit: fps
 - key: latency
   value: 0.0
   unit: ms
 - key: accuracy
   value: 0.0
   unit: f1
```
