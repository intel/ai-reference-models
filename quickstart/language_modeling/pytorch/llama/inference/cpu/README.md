<!--- 0. Title -->
# PyTorch LLAMA 7B inference (generation)

<!-- 10. Description -->
## Description

This document has instructions for running [LLAMA 7B](https://huggingface.co/decapoda-research/llama-7b-hf) inference (generation) using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Prepare model
```
  cd <clone of the model zoo>/quickstart/language_modeling/pytorch/llama/inference/cpu
  git clone https://github.com/huggingface/transformers.git
  cd transformers
  git checkout v4.28.1
  git apply ../../../../../../../models/language_modeling/pytorch/common/enable_ipex_for_transformers.diff
  pip install -e ./
  cd ..
 ```
### Model Specific Setup

* Install Intel OpenMP
  ```
  conda install intel-openmp
  ```

* Install datasets
  ```
  pip install datasets sentencepiece psutil
  ```
* Set INPUT_TOKEN before running the model
  ```
  export INPUT_TOKEN=32
  (choice in [32 64 128 256 512 1024 2016], we prefer to benchmark on 32 and 2016)
  ```

* Set OUTPUT_TOKEN before running the model
  ```
  export OUTPUT_TOKEN=32 
  (32 is preferred, while you could set any other length)
  ```

* About the BATCH_SIZE in scripts
  ```
  using BATCH_SIZE=1 for realtime mode
  using BATCH_SIZE=N for throughput mode (N could be further tuned according to the testing host, by default using 1); 
  ```

* About the BEAM_SIZE in scripts
  ```
  using BEAM_SIZE=4 by default
  ```

* Do quantization to get INT8 model before running INT8.
  ```
  #default using IPEX static quantization
  bash do_quantization.sh int8-fp32 default  # or use int8-bf16

  #optional using IPEX smoothquant for better accuracy
  bash do_quantization.sh int8-fp32 sq # or use int8-bf16
  ```

* Set ENV to use fp16 AMX if you are using a supported platform
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
  ```

# Quick Start Scripts

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash run_multi_instance_throughput.sh fp32 | bash run_multi_instance_realtime.sh fp32 | bash run_accuracy.sh fp32 |
| BF32        | bash run_multi_instance_throughput.sh bf32 | bash run_multi_instance_realtime.sh bf32 | bash run_accuracy.sh bf32 |
| BF16        | bash run_multi_instance_throughput.sh bf16 | bash run_multi_instance_realtime.sh bf16 | bash run_accuracy.sh bf16 |
| FP16        | bash run_multi_instance_throughput.sh fp16 | bash run_multi_instance_realtime.sh fp16 | bash run_accuracy.sh fp16 |
| INT8-FP32        | bash run_multi_instance_throughput.sh int8-fp32 | bash run_multi_instance_realtime.sh int8-fp32 | bash run_accuracy.sh int8-fp32 |
| INT8-BF16       | bash run_multi_instance_throughput.sh int8-bf16 | bash run_multi_instance_realtime.sh int8-bf16 | bash run_accuracy.sh int8-bf16 |

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

# Clone the Transformers repo in the llama 7b inference directory
cd <clone of the model zoo>/quickstart/language_modeling/pytorch/llama/inference/cpu
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.28.1
git apply ../../../../../../../models/language_modeling/pytorch/common/enable_ipex_for_transformers.diff
pip install -e ./
cd ..

# Env vars
export OUTPUT_DIR=<path to an output directory>

# Run a quickstart script (for example, FP32 multi-instance realtime inference)
bash run_multi_instance_realtime.sh fp32
```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)

