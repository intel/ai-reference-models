# RNN-T Training

## Description

This document has instructions for running RNN-T training using Intel-optimized PyTorch.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `training.sh` | Runs training for the specified precision (fp32, avx-fp32, bf32 or bf16). |

**Note:** The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`. 
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

|               Distributed Training                    |
|  DataType   | Throughput  
| ----------- | ----------- |
| FP32 (run 100 steps)       | bash training_multinode.sh fp32 |
| BF16        | bash training_multinode.sh bf16 |
| BF16 (run 500 steps)        | NUM_STEPS=500 bash training_multinode.sh bf16 |
| BF32        | bash training_multinode.sh bf32 |
| BF32 (run 100 steps)        | NUM_STEPS=100 bash training_multinode.sh bf32 |

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison, Torch-CCL and Jemalloc and TCmalloc

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
* Set ENV to use multi-node distributed training (no need for single-node multi-sockets)

  In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes. 
  ```bash
  export NNODES=#your_node_number
  export HOSTFILE=your_ip_list_file #one ip per line
  ```

### Run the model
Once all the setup is done, the Intel® AI Reference Models repo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory and
an output directory.

```
# Clone the Intel® AI Reference Models repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Install dependencies
./quickstart/language_modeling/pytorch/rnnt/training/cpu/install_dependency.sh

# If dataset is not dowloaded on the machine, then download and preprocess RNN-T dataset:
#Dataset takes up 60+ GB disk space. After they are decompressed, they will need 60GB more disk space. Next step is preprocessing #dataset, it will generate 110+ GB WAV file. Please make sure the disk space is enough.
export DATASET_DIR=<Where_to_save_Dataset>
./quickstart/language_modeling/pytorch/rnnt/training/cpu/download_dataset.sh

# Env vars
export OUTPUT_DIR=<path to an output directory>
export DATASET_DIR=<path to the dataset>
export PRECISION=< select from :- fp32, avx-fp32, bf16, or bf32>

# Optional environemnt variables:
export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>

# Run a quickstart script:
./quickstart/language_modeling/pytorch/rnnt/training/cpu/training.sh
```
# Run distributed training script
cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/training/cpu
bash training_multinode.sh 

# Run a quickstart script and terminate in advance
cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/training/cpu
NUM_STEPS=500 bash training.sh
```

## Attention please:
1. According to MLPerf RNN-T training README, three RNN-T datasets may need 500GB disk space to preprocess the dataset. After preprocessing done, datasets take up 110+ GB disk.

2. It will take ~4 hours to run the entire 3 datasets for one epoch in BF16. You can stop the training in advance, by adding NUM_STEPS=500 before bash script. It means, after running 500 iterations, the training process will be terminated. For now, NUM_STEPS only works for BF16 training.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
