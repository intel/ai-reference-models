# RNNT Training

RNNT Training best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Training   |    Pytorch    |       https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch        |           -           |         -          |

## Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#introduction)

## Bare Metal
### General setup

[Follow](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md) to install Miniforge and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

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

* Set ENV to use multi-node distributed training (no need for single-node multi-sockets)

  In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can [refer](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes.
  ```bash
  export NNODES=#your_node_number
  export HOSTFILE=your_ip_list_file #one ip per line
  ```

* Set ENV for model and dataset path, and optionally run with no network support

## Prepare Dataset
### Get Dataset
If dataset is not dowloaded on the machine, then download and preprocess RNN-T dataset:

Dataset takes up 60+ GB disk space. After they are decompressed, they will need 60GB more disk space. Next step is preprocessing #dataset, it will generate 110+ GB WAV file. Please make sure the disk space is enough.
```
export DATASET_DIR=<Where_to_save_Dataset>
cd models/models_v2/pytorch/rnnt/training/cpu
export MODEL_DIR=$(pwd)
./download_dataset.sh
```
## Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/rnnt/training/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation):

6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **DISTRIBUTED** (True or False)              | `export DISTRIBUTED=<True or False>`                  |
| **DATASET_DIR**             | `export DATASET_DIR=<path to rnnt_training>`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=<path to an output directory>`                               |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **profiling**(True or False)               |                               `export profiling=True`                               |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, avx-fp32, bf16, or bf32) |                              |
| **EPOCHS** (optional)    |                  `export EPOCHS=12` |                              |
| **NNODES** (required for DISTRIBUTED)              | ` export NNODES=#your_node_number`                  |
| **HOSTFILE** (required for DISTRIBUTED)              | `export HOSTFILE=#your_ip_list_file #one ip per line`                  |
| **BATCH_SIZE**(Optional)      |       `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`  |

7. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
Step time: 10.394932746887207 seconds

  1%|          | 95/17415 [17:01<48:45:46, 10.14s/it]
  1%|          | 96/17415 [17:11<48:37:28, 10.11s/it]
  1%|          | 97/17415 [17:21<48:25:47, 10.07s/it]
  1%|          | 98/17415 [17:32<49:56:43, 10.38s/it]
  1%|          | 99/17415 [17:41<48:07:52, 10.01s/it]Loss@Step: 99  ::::::: 564.8252563476562
Step time: 9.727598190307617 seconds
Done in 1071.5666544437408
total samples tested:  1280
Model training time: 835.7471186853945 s
Throughput: 1.532 fps
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
- key : throughput
  value: 1.532
  unit: fps
- key: latency
  value: 0
  unit: ms
- key: accuracy
  value: 0
  unit: AP
```
