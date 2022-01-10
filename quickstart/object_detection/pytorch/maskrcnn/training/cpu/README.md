# Mask-RCNN Training

## Description
This document has instructions for running Mask R-CNN training using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison, Torch-CCL and Jemalloc.

### Model Specific Setup

* Install dependencies
  ```
  pip install yacs opencv-python pycocotools cityscapesscripts
  conda install intel-openmp
  ```

* Install model
  ```
  cd <path to your clone of the model zoo>/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark
  python setup.py develop
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

* Set ENV to use multi-node distributed training (no need for single-node multi-sockets)

  In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes. 
  ```bash
  export NNODES=#your_node_number
  export HOSTFILE=your_ip_list_file #one ip per line
  ```

## Datasets

Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset
will be downloaded. This environment variable will be used again when running quickstart scripts.
```
cd <path to your clone of the model zoo>/quickstart/object_detection/pytorch/maskrcnn/training/cpu
export DATASET_DIR=<directory where the dataset will be saved>
bash download_dataset.sh
```

## Quick Start Scripts

|  DataType   | Throughput  |
| ----------- | ----------- |
| FP32        | bash training.sh fp32 |
| BF16        | bash training.sh bf16 |

|               stributed Training              |
|  DataType   | Throughput  |
| ----------- | ----------- |
| FP32        | bash training_multinode.sh fp32 |
| BF16        | bash training_multinode.sh bf16 |

## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have an enviornment variables set to point to the dataset directory
and an output directory.

```
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Env vars
export DATASET_DIR=<path to the COCO dataset>
export OUTPUT_DIR=<path to an output directory>

# Run a quickstart script (for example, FP32 training)
cd ${MODEL_DIR}/quickstart/object_detection/pytorch/maskrcnn/training/cpu
bash training.sh fp32

# Run distributed training script (for example, FP32 distributed training)
cd ${MODEL_DIR}/quickstart/object_detection/pytorch/maskrcnn/training/cpu/
bash training_multinode.sh fp32
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)