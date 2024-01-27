<!--- 0. Title -->
# PyTorch DistilBERT Base inference

<!-- 10. Description -->
## Description

This document has instructions for running [DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) inference using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Model Specific Setup

* Set Jemalloc and tcmalloc Preload for better performance

  The jemalloc should be built from the [General setup](#general-setup) section.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"path_to/tcmalloc/lib/libtcmalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```
* Set IOMP preload for better performance
```
  pip install packaging intel-openmp
  export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
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
  Customized BATCH_SIZE is supposed to be no larger than dataset size 872.
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
* Set ENV to use fp16 AMX if you are using a supported platform
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
  ```

* [optional] Compile model with PyTorch Inductor backend
  ```shell
  export TORCH_INDUCTOR=1
  ```

# Quick Start Scripts
|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | run_multi_instance_throughput.sh fp32 | run_multi_instance_realtime.sh fp32 | run_accuracy.sh fp32 |
| BF32        | run_multi_instance_throughput.sh bf32 | run_multi_instance_realtime.sh bf32 | run_accuracy.sh bf32 |
| BF16        | run_multi_instance_throughput.sh bf16 | run_multi_instance_realtime.sh bf16 | run_accuracy.sh bf16 |
| INT8-FP32        | run_multi_instance_throughput.sh int8-fp32 | run_multi_instance_realtime.sh int8-fp32 | run_accuracy.sh int8-fp32 |
| INT8-BF16       | run_multi_instance_throughput.sh int8-bf16 | run_multi_instance_realtime.sh int8-bf16 | run_accuracy.sh int8-bf16 |

**Note**: The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

# Datasets
Use the following instructions to download the SST-2 dataset.
Also, clone the Intel® AI Reference Models GitHub Repository and set the `MODEL_DIR` directory.
```
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)
cd -
export DATASET_DIR=$(pwd)
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip -O $DATASET_DIR/SST-2.zip
unzip $DATASET_DIR/SST-2.zip -d $DATASET_DIR/
python $MODEL_DIR/quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/convert.py $DATASET_DIR
```

# Pre-Trained Model
Follow the instructions below to download the pre-trained model. 

```
git clone https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
export FINETUNED_MODEL=$(pwd)/distilbert-base-uncased-finetuned-sst-2-english
```

## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Intel® AI Reference Models can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have an enviornment variable set to point to an output directory.

```
# Navigate to the Intel® AI Reference Models repo and set the MODEL_DIR
cd models

# Prepare model and install dependencies:
./quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/setup.sh

# Env vars
export OUTPUT_DIR=<path to an output directory>
export PRECISION=< select from :- fp32, bf32, bf16, int8-fp32, int8-bf16>
export HF_DATASETS_OFFLINE=0
export SEQUENCE_LENGTH=128 
export CORE_PER_INSTANCE=4
export FINETUNED_MODEL=<path to pre-trained model>
export DATASET_DIR=<path to dataset directory>

# Optional environemnt variables:
export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>

# Run a quickstart script
./quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/<script.sh>
```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)

