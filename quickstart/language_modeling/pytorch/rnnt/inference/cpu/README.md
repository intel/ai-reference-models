# RNN-T Inference

## Description

This document has instructions for running RNN-T inference using Intel-optimized PyTorch.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi-instance inference using 4 cores per instance for the specified precision (fp32, avx-fp32, bf32 or bf16). |
| `inference_throughput.sh` | Runs multi-instance inference using 1 instance per socket for the specified precision (fp32, avx-fp32, bf32 or bf16). |
| `accuracy.sh` | Runs an inference accuracy test for the specified precision (fp32, avx-fp32, bf32 or bf16). |

**Note:** The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`. 
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

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

### Run the model
Once all the setup is done, the Intel® AI Reference Models can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory,
an output directory and the checkpoint directory.

```
# Clone the Intel® AI Reference Models repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Install dependencies
./quickstart/language_modeling/pytorch/rnnt/inference/cpu/install_dependency_baremetal.sh

# If the dataset is not downloaded on the machine, then download and preprocess RNN-T dataset:
export DATASET_DIR=<Where_to_save_Dataset>
./quickstart/language_modeling/pytorch/rnnt/inference/cpu/download_dataset.sh

# Download pretrained model
export CHECKPOINT_DIR=#Where_to_save_pretrained_model
./quickstart/language_modeling/pytorch/rnnt/inference/cpu/download_model.sh

# Env vars
export OUTPUT_DIR=<path to an output directory>
export CHECKPOINT_DIR=<path to the pretrained model checkpoints>
export DATASET_DIR=<path to the dataset>
export PRECISION=< select from :- fp32, avx-fp32, bf16, or bf32>

# Optional environemnt variables:
export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>

# Run a quickstart script:
./quickstart/language_modeling/pytorch/rnnt/inference/cpu/<script.sh>
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)