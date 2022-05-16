<!--- 0. Title -->
# PyTorch BERT Large inference

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large SQuAD1.1 inference using
Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Prepare model
```
  cd <clone of the model zoo>/quickstart/language_modeling/pytorch/bert_large/inference/cpu
  git clone https://github.com/huggingface/transformers.git
  cd transformers
  git checkout v3.0.2
  git apply ../enable_ipex_for_squad.diff
  pip install -e ./
  cd ../

```
### Model Specific Setup
* Install dependency
```
  conda install intel-openmp
```

* Download dataset

  Please following this [link](https://github.com/huggingface/transformers/tree/v3.0.2/examples/question-answering) to get dev-v1.1.json

* Download fine-tuned model
```
  mkdir bert_squad_model
  wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
  wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
```

* Set ENV to use AMX if you are using SPR
```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```

* Set ENV for model and dataset path
```
  export FINETUNED_MODEL=#path/bert_squad_model
  export EVAL_DATA_FILE=#/path/dev-v1.1.json
```

* [optional] Do calibration to get quantization config if you want do calibration by yourself.
```
  export INT8_CONFIG=#/path/configure.json
  run_calibration.sh
```

## Quick Start Scripts

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash run_multi_instance_throughput.sh fp32 | bash run_multi_instance_realtime.sh fp32 | bash run_accuracy.sh fp32 |
| BF16        | bash run_multi_instance_throughput.sh bf16 | bash run_multi_instance_realtime.sh bf16 | bash run_accuracy.sh bf16 |
| INT8        | bash run_multi_instance_throughput.sh int8 | bash run_multi_instance_realtime.sh int8 | bash run_accuracy.sh int8 |

## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory
and an output directory.

```
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Clone the Transformers repo in the BERT large inference directory
cd quickstart/language_modeling/pytorch/bert_large/inference/cpu
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v3.0.2
git apply ../enable_ipex_for_squad.diff
pip install -e ./

# Env vars
export FINETUNED_MODEL=<path to the fine tuned model>
export EVAL_DATA_FILE=<path to dev-v1.1.json file>
export OUTPUT_DIR=<path to an output directory>

# Run a quickstart script (for example, FP32 multi-instance realtime inference)
bash run_multi_instance_realtime.sh fp32
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

