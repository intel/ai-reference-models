# RNN-T Inference

## Description

This document has instructions for running RNN-T inference using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison and Jemalloc.

### Model Specific Setup
* Install dependencies
  ```bash
  export MODEL_DIR=<path to your clone of the model zoo>
  bash ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu/install_dependency_baremetal.sh
  ```

* Download and preprocess RNN-T dataset:
  ```bash
  export DATASET_DIR=#Where_to_save_Dataset
  bash ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu/download_dataset.sh
  ```

* Download pretrained model
  ```bash
  export CHECKPOINT_DIR=#Where_to_save_pretrained_model
  bash ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu/download_model.sh
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

# Quick Start Scripts

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash inference_throughput.sh fp32 | bash inference_realtime.sh fp32 | bash accuracy.sh fp32 |
| BF16        | bash inference_throughput.sh bf16 | bash inference_realtime.sh bf16 | bash accuracy.sh bf16 |

## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory,
an output directory and the checkpoint directory.

```
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Env vars
export OUTPUT_DIR=<path to an output directory>
export CHECKPOINT_DIR=<path to the pretrained model checkpoints>
export DATASET_DIR=<path to the dataset>

# Run a quickstart script (for example, FP32 batch inference)
cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu
bash inference_throughput.sh fp32
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)