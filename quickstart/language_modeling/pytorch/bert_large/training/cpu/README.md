<!--- 0. Title -->
# PyTorch BERT Large training
<!-- 10. Description -->
## Description

This document has instructions for running BERT Large pre-training using
Intel-optimized PyTorch.

## Datasets

BERT Large training uses the config file and enwiki-20200101 dataset from the
[MLCommons training GitHub repo](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert).

Follow the instructions in their documentation to download the files and
preprocess the dataset to create TF records files. Set the `DATASET_DIR`
environment variable to the path to the TF records directory. Your directory
should look similar like this:
```
<DATASET_DIR>
├── seq_128
│   └── part-00000-of-00500_128
└── seq_512
    └── part-00000-of-00500
```


## Bare Metal

### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Model Specific Setup
* Install dependence
```
  pip install datasets accelerate tfrecord
  conda install openblas
  conda install faiss-cpu -c pytorch
  conda install intel-openmp
```

* Prepare transformers code
```
  git clone https://github.com/huggingface/transformers.git
  cd transformers
  git checkout v4.11.3
  wget https://github.com/huggingface/transformers/pull/13714.diff
  git apply 13714.diff
  pip install -e ./
  cd ../

```

* Set ENV to use AMX if you are using SPR
```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```

* Set ENV for model and dataset path
```
  export DATASET_DIR=/path/to/dataset/tfrecord_dir
  export TRAIN_SCRIPT=/path/to/models/language_modeling/pytorch/bert_large/training/run_pretrain_mlperf.py

  ###for phase 1 get from https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT
  export BERT_MODEL_CONFIG=/path/to/bert_config.json

  ###for phase 2
  export PRETRAINED_MODEL=/path/to/bert_large_mlperf_checkpoint/checkpoint/
```

### Training CMD

|  DataType   | Phase 1  |  Phase 2 |
| ----------- | ----------- | ----------- |
| FP32        | bash run_bert_pretrain_phase1.sh fp32 | bash run_bert_pretrain_phase2.sh fp32 |
| BF16        | bash run_bert_pretrain_phase1.sh bf16 | bash run_bert_pretrain_phase2.sh bf16 |


<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

