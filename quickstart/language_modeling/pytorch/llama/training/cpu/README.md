<!--- 0. Title -->
# PyTorch LLAMA2 7B lora apalca finetuning training

<!-- 10. Description -->
## Description

This document has instructions for running [LLaMA2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)  lora apalca finetuning using Intel-optimized PyTorch.

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

* Set ENV to use multi-nodes distributed training (no need for single-node multi-sockets)

In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes. 
```
export NNODES=#your_node_number (default using 4 nodes, 8 sockets)
# create your_ip_list_file, one ip per line, like (or self edit):
scontrol show hostname > ./hostfile

export HOSTFILE=hostfile 

# [Optional] The following is needed if you have not set torch ccl and oneccl
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.torch-ccl.git
cd frameworks.ai.pytorch.torch-ccl
git checkout public_master
git submodule sync
git submodule update --init --recursive
python setup.py install
cd ../

git clone https://github.com/oneapi-src/oneCCL.git
cd oneCCL
mkdir build
cd build
cmake ..
make -j install
source _install/env/setvars.sh
cd ../..

```

# Quick Start Scripts  (single socket)

|  DataType   | Throughput  |
| ----------- | ----------- |
| BF16        | bash run_lora_finetune.sh bf16  |
| FP16        | bash run_lora_finetune.sh fp16  |
| FP32        | bash run_lora_finetune.sh fp32  |
| BF32        | bash run_lora_finetune.sh bf32  |

# Quick Start Scripts  (distributed)
|  DataType   | Throughput  |
| ----------- | ----------- |
| BF16        | bash run_lora_finetune_ddp.sh bf16  |
| FP16        | bash run_lora_finetune_ddp.sh fp16  |
| FP32        | bash run_lora_finetune_ddp.sh fp32  |
| BF32        | bash run_lora_finetune_ddp.sh bf32  |

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
git apply ../../../../../../../models/language_modeling/pytorch/common/enable_ipex_for_transformers.diff
pip install -e ./
cd ..

#[optional] you may need to get access to llama2 weights from HF
Apply the access in this page [LLaMA2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) with your huggingface account
huggingface-cli login
{your huggingface token}


# Env vars
export OUTPUT_DIR=<path to an output directory>

# Run a quickstart script (for example, bf16 training)
bash run_lora_finetune.sh bf16
```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)

