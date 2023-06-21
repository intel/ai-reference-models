<!--- 0. Title -->
# PyTorch DLRM inference

<!-- 10. Description -->
## Description

This document has instructions for running torchrec DLRM inference using
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

## Datasets
The dataset can be downloaded and preprocessed by following https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm#create-the-synthetic-multi-hot-dataset.
We also provided a preprocessed scripts based on the instruction above. `preprocess_raw_dataset.sh`.
After you loading the raw dataset `day_*.gz` and unzip them to RAW_DIR.
```bash
export MODEL_DIR=<where you clone this repo>
export RAW_DIR=<the unziped raw dataset>
export TEMP_DIR=<where your choose the put the temp file during preprocess>
export PREPROCESSED_DIR=<where your choose the put the one-hot dataset>
export MULTI_HOT_DIR=<where your choose the put the multi-hot dataset>
bash preprocess_raw_dataset.sh
```

## Pre-Trained checkpoint
Your can download and unzip checkpoint by following
https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch#downloading-model-weights


## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_performance.sh` | Run inference to verify performance for the specified precision (fp32, bf32, bf16, fp16, or int8). |
| `test_accuracy.sh` | Run inference to verify auroc for the specified precision (fp32, bf32, bf16, fp16, or int8). |

## Run the model

```bash
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)
export OUTPUT_DIR=<specify the log dir to save log>

# Env vars
export PRECISION=<specify the precision to run>

# Run a quickstart script for bare metal performance)
cd ${MODEL_DIR}/quickstart/recommendation/pytorch/dlrm/inference/cpu
bash inference_performance.sh


# Run a quickstart script for accuracy test
cd ${MODEL_DIR}/quickstart/recommendation/pytorch/dlrm/inference/cpu
export DATASET_DIR=<multi-hot dataset dir>
export WEIGHT_DIR=<offical released checkpoint>
bash test_accuracy.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
