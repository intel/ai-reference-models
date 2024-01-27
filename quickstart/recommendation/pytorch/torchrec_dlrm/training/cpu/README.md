<!--- 0. Title -->
# PyTorch DLRM training

<!-- 10. Description -->
## Description

This document has instructions for running torchrec DLRM training using
Intel-optimized PyTorch for bare metal.

## Bare Metal
### General Setup
Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, and Jemalloc.

### Model Specific Setup

* Install dependencies
  ```bash
  cd <clone of the model zoo>/quickstart/recommendation/pytorch/torchrec_dlrm
  pip install requirements.txt
  ```

* Set Jemalloc Preload for better performance

  The jemalloc should be built from the [General setup](#general-setup) section.
  ```bash
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto"
  ```

* Set IOMP preload for better performance

  IOMP should be installed in your conda env from the [General setup](#general-setup) section.
  ```bash
  export LD_PRELOAD=<path to the intel-openmp directory>/lib/libiomp5.so:$LD_PRELOAD
  ```

* Set ENV to use AMX if you are using SPR
  ```bash
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

* [optional] Compile model with PyTorch Inductor backend
  ```shell
  export TORCH_INDUCTOR=1
  ```

## Datasets

Use random dataset.

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training_performance.sh` | Run training to verify performance for the specified precision (fp32, bf32, fp16, or bf16). |

## Run the model

```bash
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)
export OUTPUT_DIR=<specify the log dir to save log>
# Env vars
export PRECISION=<specify the precision to run>

# Run a quickstart script (for example, bare metal performance)
cd ${MODEL_DIR}/quickstart/recommendation/pytorch/torchrec_dlrm/training/cpu
# we recommend to use real dataset to evaluate performance data since the performance will highly related with input data distribution
# do not set ONE_HOT_DATASET_DIR will use dummy data
export ONE_HOT_DATASET_DIR=<your one hot dataset dir>
# Normal Training
bash training_performance.sh
# Distributed training
DIST=1 bash training_performance.sh
# Convergence test for Normal Training
CONVERGENCE=1 bash training_performance.sh
# Convergence test for Distributed Training
DIST=1 CONVERGENCE=1 bash training_performance.sh
```

## Distributed training advance setting
The default behaviour for `DIST=1 bash training_performance.sh`
(1) Will run 2 instance on numa node `0, 1` within non-SNC mode.
(2) Will run n instance on socket 0 on sub numa node `0, 1, ..., n` with SNC-N mode. For example, will run 3 instance on socket 0, sub numa node `0, 1, 2` with SNC3.
(3) You can further setting it by setting `NODE_LIST` and `NODE` and `PROC_PER_NODE`. For example, if you wish to run 4 instance on 2 nodes, and run on numa nodes `2, 3` for each nodes
You can run
```bash
export NODE_LIST=2,3
export NODE=2
export PROC_PER_NODE=2
DIST=1 bash training_performance.sh
```
Please remember to config your IP address in `hostfile1`, 1 address per line.


<!--- 80. License -->
## License

[LICENSE](/LICENSE)
