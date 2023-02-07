<!--- 0. Title -->
# PyTorch DistilBERT Base inference

<!-- 10. Description -->
## Description

This document has instructions for running [DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) inference using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Prepare model
```
  cd <clone of the model zoo>/quickstart/language_modeling/pytorch/distilbert_base/inference/cpu
  git clone https://github.com/huggingface/transformers.git
  cd transformers
  git checkout v4.18.0
  git apply ../enable_ipex_for_distilbert-base.diff
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

* Set SEQUENCE_LENGTH before running the model
  ```
  export SEQUENCE_LENGTH=128 
  (128 is preferred, while you could set any other length)
  ```

* Set CORE_PER_INSTANCE before running realtime mode
  ```
  export CORE_PER_INSTANCE=4
  (4cores per instance setting is preferred, while you could set any other config like 1core per instance)
  ```

* About the BATCH_SIZE in scripts
  ```
  Throughput mode is using BATCH_SIZE=[4 x core number] by default in script (which could be further tuned according to the testing host); 
  Realtime mode is using BATCH_SIZE=[1] by default in script; 
  
  Note: If you would have a SPR-56C host, BATCH_SIZE=205 is perferred for INT8-BF16 Throughput mode and BATCH_SIZE=198 is perferred for BF16 Throughput mode.
  ```

* Do calibration to get quantization config before running INT8 (Default attached is produced with sequence length 128).
  ```
  #Set the SEQUENCE_LENGTH to which is going to run when doing the calibration.
  bash do_calibration.sh
  ```
* [Optional for offline tests] Prepare model and dataset files locally
  ```
  (1) download model and sst2 dataset (make sure to install git-lfs first by apt-get install git-lfs)
  bash download_model_dataset.sh
  #by default they are downloaded in current path
  #note that you should do this after you prepared model (transformers repo)

  (2) make following changes in the scirpts to run:
  delete: --task_name sst2  ==>  add: --train_file {path/to/data_file}/SST-2/train.csv --validation_file {path/to/data_file}/SST-2/dev.csv 
  
  (3) export model path
  export FINETUNED_MODEL={path/to/model_file}/distilbert-base-uncased-finetuned-sst-2-english
  
  (4) run scirpt with HF_DATASETS_OFFLINE=1 flag, like:
  HF_DATASETS_OFFLINE=1 bash run_multi_instance_throughput.sh fp32
  
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

