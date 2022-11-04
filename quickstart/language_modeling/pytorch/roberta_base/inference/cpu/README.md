<!--- 0. Title -->
# PyTorch RoBERTa Base inference

<!-- 10. Description -->
## Description

This document has instructions for running [RoBERTa Base SQuAD 2.0](https://huggingface.co/deepset/roberta-base-squad2) inference using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Prepare model
```
  cd <clone of the model zoo>/quickstart/language_modeling/pytorch/roberta_base/inference/cpu
  git clone https://github.com/huggingface/transformers.git
  cd transformers
  git checkout v4.10.0
  git apply ../enable_ipex_for_roberta-base.diff
  pip install -e ./
  cd ../
   
```
### Model Specific Setup
* Install Intel OpenMP
  ```
  conda install intel-openmp
  ```

* Install datasets
  ```
  pip install datasets
  ```

* Set ENV to use AMX if you are using SPR
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

## Quick Start Scripts

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash run_multi_instance_throughput.sh fp32 | bash run_multi_instance_realtime.sh fp32 | bash run_accuracy.sh fp32 |
| BF16        | bash run_multi_instance_throughput.sh bf16 | bash run_multi_instance_realtime.sh bf16 | bash run_accuracy.sh bf16 |

## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have an enviornment variable set to point to an output directory.

```
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Clone the Transformers repo in the RoBERTa Base inference directory
cd quickstart/language_modeling/pytorch/roberta_base/inference/cpu
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.10.0
git apply ../enable_ipex_for_roberta-base.diff
pip install -e ./
cd ../

# Env vars
export OUTPUT_DIR=<path to an output directory>

# Run a quickstart script (for example, FP32 multi-instance throughput)
bash run_multi_instance_throughput.sh fp32
```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)

