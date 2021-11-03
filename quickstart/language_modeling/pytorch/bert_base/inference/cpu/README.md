<!--- 0. Title -->
# PyTorch BERT Base inference

<!-- 10. Description -->
## Description

This document has instructions for running [BERT Base SQuAD1.1](https://huggingface.co/csarron/bert-base-uncased-squad-v1) inference using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Prepare model
```
  cd <clone of the model zoo>/quickstart/language_modeling/pytorch/bert_base/inference/cpu
  git clone https://github.com/huggingface/transformers.git
  cd transformers
  git checkout v4.10.0
  git apply ../enable_ipex_for_bert-base.diff
  pip install -e ./
  cd ../
   
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

* Set ENV to use AMX if you are using SPR
```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```
* Set the output path
```
  export OUTPUT_DIR=/path/to/save/output
```


### Inference CMD

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash run_multi_instance_throughput.sh fp32 | bash run_multi_instance_realtime.sh fp32 | bash run_accuracy.sh fp32 |
| BF16        | bash run_multi_instance_throughput.sh bf16 | bash run_multi_instance_realtime.sh bf16 | bash run_accuracy.sh bf16 |



<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)

