<!--- 0. Title -->
# PyTorch DLRM training

<!-- 10. Description -->
## Description

This document has instructions for running DLRM training using
Intel-optimized PyTorch for bare metal.

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training.sh` | Run training for the specified precision (fp32, avx-fp32, bf32 or bf16). |
| `test_convergency.sh` | Run fully convergency test for the specified precision (fp32, bf16, bf32). |
| `distribute_training.sh` | Run distribute training on 1 node with 2 sockets for the specified precision (fp32, bf16, bf32). |

**Note**: The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

## Datasets

### Criteo Terabyte Dataset

The [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) is
used to run DLRM. To download the dataset, you will need to visit the Criteo website and accept
their terms of use:
[https://labs.criteo.com/2013/12/download-terabyte-click-logs/](https://labs.criteo.com/2013/12/download-terabyte-click-logs/).
Copy the download URL into the command below as the `<download url>` and
replace the `<dir/to/save/dlrm_data>` to any path where you want to download
and save the dataset.
```bash
export DATASET_DIR=<dir/to/save/dlrm_data>

mkdir ${DATASET_DIR} && cd ${DATASET_DIR}
curl -O <download url>/day_{$(seq -s , 0 23)}.gz
gunzip day_*.gz
```
The raw data will be automatically preprocessed and saved as `day_*.npz` to
the `DATASET_DIR` when DLRM is run for the first time. On subsequent runs, the
scripts will automatically use the preprocessed data.

### General Setup
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

## Run the model

Follow the instructions above to setup your bare metal environment, do the
model-specific setup and download and prepropcess the datsaet. Once all the
setup is done, the Intel® AI Reference Models can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory,
precision, and an output directory. The `NUM_BATCH` environment variable
can be set to specify the number of batches to run.

```bash
# Clone the Intel® AI Reference Models repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Install dependencies
pip install quickstart/recommendation/pytorch/dlrm/requirements.txt

# Env vars
export PRECISION=<specify the precision to run: fp32, bf16 or bf32>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export NUM_BATCH=<10000 for test performance and 50000 for testing convergence trend>  

# Optional environemnt variables:
export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>

# [optional] Compile model with PyTorch Inductor backend 
export TORCH_INDUCTOR=1

# Run quick start script:
./quickstart/recommendation/pytorch/dlrm/training/cpu/training.sh

# Or, run quickstart script for testing fully convergency
# Navigate to the DLRM training quickstart directory
cd ${MODEL_DIR}/quickstart/recommendation/pytorch/dlrm/training/cpu
bash test_convergence.sh

# Run quickstart to distribute training dlrm on 2 sockets
# Note, you need to follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Torch-CCL and run this command on the machine which sockets larger than 2
# The default setting will run 2 ranks on 1 nodes with BATCH_SIZE=32768 and CCL_WORKER_COUNT=8
NUM_BATCH=10000 bash distribute_training.sh

# To run more settings, you can config following ENV
export BATCH_SIZE=65536
export NUM_CCL_WORKER=4
export HOSTFILE=<your hostfile>
export NODE=2
NUM_BATCH=10000 bash distribute_training.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
