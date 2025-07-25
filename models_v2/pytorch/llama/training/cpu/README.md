<!--- 0. Title -->
# PyTorch LLAMA2 7B lora apalca finetuning training

<!-- 10. Description -->
## Description

This document has instructions for running [LLaMA2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)  lora apalca finetuning using Intel-optimized PyTorch.

## Bare Metal
### General setup

[Follow](/docs/general/pytorch/BareMetalSetup.md) to install and build Pytorch, IPEX, TorchVison and TCMalloc.

### Model Specific Setup

* Install Intel OpenMP
  ```
  pip install packaging intel-openmp accelerate
  ```
* Set IOMP and tcmalloc Preload for better performance
  ```
  export LD_PRELOAD="<path_to>/tcmalloc/lib/libtcmalloc.so":"<path_to_iomp>/lib/libiomp5.so":$LD_PRELOAD
  ```

* Set ENV to use multi-nodes distributed training (no need for single-node multi-sockets)

In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can [refer](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes.
```
export NNODES=#your_node_number (default using 1 node)
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

# Dataset
  ```
  # Get the dataset here: https://github.com/tloen/alpaca-lora/blob/main/alpaca_data.json
  wget https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data.json
  mv alpaca_data.json <clone of the AI Reference models>/models_v2/pytorch/llama/training/cpu

  # Get the dataset template here: https://github.com/tloen/alpaca-lora/blob/main/templates/alpaca.json
  wget https://raw.githubusercontent.com/tloen/alpaca-lora/main/templates/alpaca.json
  mkdir <clone of the AI Reference models>/models_v2/pytorch/llama/training/cpu/templates
  mv alpaca.json <clone of the AI Reference models>/models_v2/pytorch/llama/training/cpu/templates
  ```

# Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/llama/training/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation)

6. #[optional] you may need to get access to llama2 weights from HF
    Apply the access in this page [LLaMA2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) with your huggingface account
    huggingface-cli login
    {your huggingface token}

7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **DDP**                    | `export DDP=False (True or False)`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=<path to an output directory>`                               |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, bf32, bf16, fp16) |
| **MODEL_DIR**               |        `export MODEL_DIR=$(pwd)`                               |
| **BATCH_SIZE** (optional)    |          `export BATCH_SIZE=256`                                |
| **NNODES** (Optional)     |                 `export NNODES=1`                                |

## Output

Single-tile output will typically looks like:

```
2024-05-17 22:35:31,097 - root - INFO - ---------- Summary: ----------
2024-05-17 22:35:31,097 - root - INFO - inference-latency: 18.211 sec.
2024-05-17 22:35:31,097 - root - INFO - first-token-latency: 4.227 sec.
2024-05-17 22:35:31,097 - root - INFO - rest-token-latency: 0.110 sec.
2024-05-17 22:35:31,097 - root - INFO - P90-rest-token-latency: 0.111 sec.
2024-05-17 22:35:36,648 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;total-latency;bf16;1; 18.179000
2024-05-17 22:35:36,655 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;first-token-latency;bf16;1; 4.238500
2024-05-17 22:35:36,664 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;rest-token-latency;bf16;1; 0.110000
2024-05-17 22:35:36,671 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;P90-rest-token-latency;bf16;1; 0.110500
2024-05-17 22:35:36,678 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;token_per_sec;bf16;1; 9.110
2024-05-17 22:35:36,686 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;first_token_thp;bf16;1; 0.236
```
Final results of the inference run can be found in `results.yaml` file.
```
results:
- key: first token throughput
  value: 15.648000
- key: rest token throughput
  value: 0.284250
- key: first token latency
  value: 4.238500
- key: rest_token_latency
  value: 0.110000
- key: accuracy
  value: 93.17
```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
