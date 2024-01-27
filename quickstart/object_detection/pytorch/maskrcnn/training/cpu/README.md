# Mask-RCNN Training

## Description
This document has instructions for running Mask R-CNN training using Intel-optimized PyTorch.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `training.sh` | Runs training for the specified precision (fp32, avx-fp32，bf16, or bf32). |

> Note: The `avx-fp32` precisions run the same scripts as `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

For Distributed Training:
|  DataType   | Throughput  |
| ----------- | ----------- |
| FP32        | bash training_multinode.sh fp32 |
| BF16        | bash training_multinode.sh bf16 |
| BF32        | bash training_multinode.sh bf32 |

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison and Jemalloc and TCmalloc

### Model Specific Setup
* Set Jemalloc and tcmalloc Preload for better performance

  The jemalloc should be built from the [General setup](#general-setup) section.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"path_to/tcmalloc/lib/libtcmalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```

* Set IOMP preload for better performance

  IOMP should be installed in your conda env from the [General setup](#general-setup) section.
  ```
  pip install packaging intel-openmp
  export LD_PRELOAD=<path to the intel-openmp directory>/lib/libiomp5.so:$LD_PRELOAD
  ```

* Follow the instructions to setup your bare metal environment on either Linux or Windows systems. Once all the setup is done,
  the Intel® AI Reference Models can be used to run a [quickstart script](#quick-start-scripts).
  Ensure that you have a clone of the [Intel® AI Reference Models Github repository](https://github.com/IntelAI/models) and navigate to the directory.
  ```
  git clone https://github.com/IntelAI/models.git
  cd models
  ```
* Install model
  ```
  python models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/setup.py develop
  ```
* Datasets

  Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
  Export the `DATASET_DIR` environment variable to specify the directory where the dataset
  will be downloaded. This environment variable will be used again when running quickstart scripts.
  ```
  cd quickstart/object_detection/pytorch/maskrcnn/training/cpu
  export DATASET_DIR=<directory where the dataset will be saved>
  bash download_dataset.sh
  cd - 
  ```
* Set ENV to use multi-node distributed training (no need for single-node multi-sockets)

  In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes. 
  ```bash
  export NNODES=#your_node_number
  export HOSTFILE=your_ip_list_file #one ip per line
  ```

## Run the model
Once all the above setup is done,we can run a [quickstart script](#quick-start-scripts).
Ensure that you have an enviornment variables set to point to the dataset directory
and an output directory.

```
# Make sure you are inside the Intel® AI Reference Models directory
export MODEL_DIR=$(pwd)

# Set the environment variable:
export DATASET_DIR=<path to the COCO dataset>
export OUTPUT_DIR=<path to an output directory>
export PRECISION=< select from :- fp32, avx-fp32, bf16, or bf32>

# Optional environemnt variables:
export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>

# Install dependency:
./quickstart/object_detection/pytorch/maskrcnn/training/cpu/setup.sh

# Run a quickstart script:
./quickstart/object_detection/pytorch/maskrcnn/training/cpu/<quickstart_script.sh>

# Run distributed training script (for example, FP32 distributed training)
cd ${MODEL_DIR}/quickstart/object_detection/pytorch/maskrcnn/training/cpu/
export LOCAL_BATCH_SIZE=#local batch_size
bash training_multinode.sh fp32
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)