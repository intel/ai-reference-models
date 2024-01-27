# SSD-ResNet34 Training

## Description
This document has instructions for running SSD-ResNet34 training using Intel-optimized PyTorch.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `throughput.sh` | Tests the training performance for SSD-ResNet34 for the specified precision (fp32, avx-fp32, bf32 or bf16). |
| `accuracy.sh` | Tests the training accuracy for SSD-ResNet34 for the specified precision (fp32, avx-fp32, bf32 or bf16). |

**Note**: The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

|           Distributed Training          |
|  DataType   | Throughput  |   Accuracy  |
| ----------- | ----------- | ----------- |
| FP32        | bash throughput_dist.sh fp32 | bash accuracy_dist.sh fp32 |
| BF16        | bash throughput_dist.sh bf16 | bash accuracy_dist.sh bf16 |
| BF32        | bash throughput_dist.sh bf32 | bash accuracy_dist.sh bf32 |

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison, Torch-CCL TCmalloc and Jemalloc.

### Model Specific Setup
* Set Jemalloc and tcmalloc Preload for better performance

  The jemalloc should be built from the [General setup](#general-setup) section.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"path_to/tcmalloc/lib/libtcmalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```

* Set IOMP preload for better performance

  IOMP should be installed in your conda/virtual env from the [General setup](#general-setup) section.
  ```
  pip install packaging intel-openmp
  export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
  ```

* Set ENV to use multi-node distributed training (no need for single-node multi-sockets)

  In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes.
  ```
  export NNODES=#your_node_number
  export HOSTFILE=your_ip_list_file #one ip per line
  ```

### Run the model

Once all the setup is done, the Intel® AI Reference Models repo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have an enviornment variables set to point to the dataset directory
and an output directory.

```
# Clone the Intel® AI Reference Models riepo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Dataset
Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset
will be downloaded. This environment variable will be used again when running quickstart scripts.

cd quickstart/object_detection/pytorch/ssd-resnet34/training/cpu
export DATASET_DIR=<directory where the dataset will be saved>
bash download_dataset.sh
cd - 
cd ${MODEL_DIR}

# install model specific dependencies
./quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/setup.sh

# Download pretrained model
export CHECKPOINT_DIR=<directory where the pretrained model will be saved>
bash quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/download_model.sh

# Env vars
export DATASET_DIR=<path to the COCO dataset>
export OUTPUT_DIR=<path to an output directory>
export PRECISION=<select from :- fp32, avx-fp32, bf16, or bf32>

# Optional environemnt variables:
export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>

# Run a quickstart script
./quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/<script.sh>
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
