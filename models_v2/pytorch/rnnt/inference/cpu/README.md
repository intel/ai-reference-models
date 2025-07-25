# RNNT Inference

RNNT Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    Pytorch    |       https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch        |           -           |         -          |

## Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#introduction)

## Bare Metal
### General setup

[Follow](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md) to install Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

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

## Prepare Dataset
### Get Dataset
If dataset is not dowloaded on the machine, then download and preprocess RNN-T dataset:

Dataset takes up 60+ GB disk space. After they are decompressed, they will need 60GB more disk space. Next step is preprocessing #dataset, it will generate 110+ GB WAV file. Please make sure the disk space is enough.
```
export DATASET_DIR=<Where_to_save_Dataset>
cd models/models_v2/pytorch/rnnt/inference/cpu
export MODEL_DIR=$(pwd)
./download_dataset.sh
```

### Get Pretrained Model
```
cd $MODEL_DIR
./download_model.sh
```

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/rnnt/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation)

6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)              | `export TEST_MODE=THROUGHPUT`                  |
| **DATASET_DIR**             | `export DATASET_DIR=<path to rnnt_dataset>`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=<path to an output directory>`                               |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, avx-fp32, bf16, or bf32) |                              |
| **CHECKPOINT_DIR**     |                  `export CHECKPOINT_DIR=<path to the pretrained model checkpoints>` |                              |
| **BATCH_SIZE**(Optional)      |       `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`  |

7. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
Evaluation WER: 0.07326936509687144
Accuracy: 0.926730634903129
P99 Latency 13966.99 ms
total samples tested:  2703
total time (encoder + decoder, excluded audio processing):  29.09852635115385 s
dataset size:  2703
Throughput: 92.891 fps


=========================>>>>>>
Evaluation WER: 0.07326936509687144
Accuracy: 0.926730634903129
P99 Latency 14268.15 ms
total samples tested:  2703
total time (encoder + decoder, excluded audio processing):  29.805503878742456 s
dataset size:  2703
Throughput: 90.688 fps
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
- key : throughput
  value: 91.7895
  unit: fps
- key: latency
  value: 14,117.57
  unit: ms
- key: accuracy
  value: 0.927
  unit: AP
```
