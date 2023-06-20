<!--- 0. Title -->
# PyTorch Bloom 1B4 inference (generation)

<!-- 10. Description -->
## Description

This document has instructions for running [Bloom 1B4](https://huggingface.co/Langboat/bloom-1b4-zh) inference (generation) using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Prepare model
```
  cd <clone of the model zoo>/quickstart/language_modeling/pytorch/bloom/inference/cpu
  git clone https://github.com/huggingface/transformers.git
  cd transformers
  git checkout v4.28.1
  pip install -r requirements.txt
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
  pip install datasets
  ```
* Set INPUT_TOKEN before running the model
  ```
  export INPUT_TOKEN=1024
  (choice in [32 512 1024])
  ```


* Set OUTPUT_TOKEN before running the model
  ```
  export OUTPUT_TOKEN=128 
  (128 is preferred, while you could set any other length)
  ```

* Set CORE_PER_INSTANCE before running realtime mode
  ```
  export CORE_PER_INSTANCE=4
  (4cores per instance setting is preferred, while you could set any other config like 1core per instance)
  ```

* About the BATCH_SIZE in scripts
  ```
  using BATCH_SIZE=1 by default in scripts (which could be further tuned according to the testing host); 
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

# Clone the Transformers repo in the bloom 1b4 inference directory
cd <clone of the model zoo>/quickstart/language_modeling/pytorch/bloom/inference/cpu
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.28.1
pip install -r requirements.txt
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

