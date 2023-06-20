<!--- 0. Title -->
# PyTorch LLAMA 7B lora apalca finetuning training (single socket)

<!-- 10. Description -->
## Description

This document has instructions for running [LLAMA 7B](https://huggingface.co/decapoda-research/llama-7b-hf) lora apalca finetuning using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Prepare dependency
```
  pip install -r requirements.txt
 ```
### Model Specific Setup

* Install Intel OpenMP
  ```
  conda install intel-openmp
  ```

# Quick Start Scripts

|  DataType   | Throughput  |
| ----------- | ----------- |
| BF16        | bash run_lora_finetune.sh bf16  |
| FP16        | bash run_lora_finetune.sh fp16  |
| FP32        | bash run_lora_finetune.sh fp32  |
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
cd <clone of the model zoo>/quickstart/language_modeling/pytorch/llama/training/cpu
pip uninstall transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.28.1
pip install -r requirements.txt
git apply ../../../../../../../models/language_modeling/pytorch/common/enable_ipex_for_transformers.diff
pip install -e ./
cd ..

# Env vars
export OUTPUT_DIR=<path to an output directory>

# Run a quickstart script (for example, bf16 training)
bash run_lora_finetune.sh bf16
```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)

