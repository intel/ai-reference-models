<!--- 0. Title -->
# PyTorch DistilBERT Base inference

<!-- 10. Description -->
## Description

This document has instructions for running [DistilBERT Base SQuAD1.1](https://huggingface.co/distilbert-base-uncased-distilled-squad) inference using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Model Specific Setup

* Install Intel OpenMP
  ```
  conda install intel-openmp
  ```

* Install datasets
  ```
  pip install datasets
  ```

* Set ENV to use AMX if you are using SPR an linux kernel < 5.16
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```
* Set the following batch size if you are using SPR-56core and running bf16 or int8-bf16 for throughput mode in "run_multi_instance_throughput.sh"
  ```
  bf16:
  BATCH_SIZE=${BATCH_SIZE:-198}
  int8-bf16:
  BATCH_SIZE=${BATCH_SIZE:-168}
  (Other conditions can use [4 x core number] by default)
  ```

* [optional] Do calibration to get quantization config if you want do calibration by yourself.
  ```
  bash do_calibration.sh
  ```

# Quick Start Scripts

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash run_multi_instance_throughput.sh fp32 | bash run_multi_instance_realtime.sh fp32 | bash run_accuracy.sh fp32 |
| BF16        | bash run_multi_instance_throughput.sh bf16 | bash run_multi_instance_realtime.sh bf16 | bash run_accuracy.sh bf16 |
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

# Clone the Transformers repo in the DistilBERT Base inference directory
cd quickstart/language_modeling/pytorch/distilbert_base/inference/cpu
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.18.0
git apply ../enable_ipex_for_distilbert-base.diff
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

