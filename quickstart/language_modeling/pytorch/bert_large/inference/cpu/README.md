<!--- 0. Title -->
# PyTorch BERT Large inference

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large SQuAD1.1 inference using
Intel-optimized PyTorch.

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

* Set ENV to use fp16 AMX if you are using a supported platform
```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```

* Set ENV for model and dataset path, and optionally run with no network support
```
  export FINETUNED_MODEL=#path/bert_squad_model
  export EVAL_DATA_FILE=#/path/dev-v1.1.json
  
  
  ### [optional] Pure offline mode to benchmark:
  change --tokenizer_name to #path/bert_squad_model in scripts before running
  e.g. --tokenizer_name ${FINETUNED_MODEL} in run_multi_instance_throughput.sh
  
```

* [optional] Do calibration to get quantization config if you want do calibration by yourself.
```
  export INT8_CONFIG=#/path/configure.json
  run_calibration.sh
```

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `run_multi_instance_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16) using the [huggingface fine tuned model](https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin). |
| `run_multi_instance_throughput.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, avx-fp32, int8, avx-int8,bf32 or bf16) using the [huggingface fine tuned model](https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin). |
| `run_accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16) using the [huggingface fine tuned model](https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin). |

**Note**: The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

## Quick Start Scripts for fast_bert with TPP optimization 

|  DataType   |  Accuracy and Throughput  |
| ----------- |  ----------- |
| BF16        | bash fast_bert_squad_infer.sh --use_tpp --unpad --tpp_bf16|

## Datasets
Please following this [link](https://github.com/huggingface/transformers/tree/v3.0.2/examples/question-answering) to get `dev-v1.1.json` and set the `EVAL_DATA_FILE` environment variable to point to the file:
```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
export EVAL_DATA_FILE=$(pwd)/dev-v1.1.json
```
## Pre-Trained Model
Download the `config.json` and fine tuned model from huggingface and set the `FINETUNED_MODEL` environment variable to point to the directory that has both files:
```
mkdir bert_squad_model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
export FINETUNED_MODEL=$(pwd)/bert_squad_model
```

## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Intel® AI Reference Models can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory
and an output directory.

```
# Clone the Intel® AI Reference Models repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Clone the Transformers repo in the BERT large inference directory
./quickstart/language_modeling/pytorch/bert_large/inference/cpu/setup.sh

# Env vars
export FINETUNED_MODEL=<path to the fine tuned model>
export EVAL_DATA_FILE=<path to dev-v1.1.json file>
export OUTPUT_DIR=<path to an output directory>
export PRECISION=< select from :- fp32, bf32, bf16, int8, avx-int8, avx-fp32>

# Optional environemnt variables:
export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>

# Run a quickstart script:
./quickstart/language_modeling/pytorch/bert_large/inference/cpu/<script.sh>
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

