# DLRM v2 Inference

DLRM v2 Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm       |           -           |         -          |

# Pre-Requisite
## Bare Metal
### General setup

Follow [link](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md) to build Pytorch, IPEX, TorchVison and TCMalloc.

### Model Specific Setup

* Installation of [Build PyTorch + IPEX + TorchVision Jemalloc and TCMalloc](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md)
* Installation of [oneccl-bind-pt](https://pytorch-extension.intel.com/release-whl/stable/cpu/us/oneccl-bind-pt/) (if running distributed)
* Set Jemalloc and tcmalloc Preload for better performance

  The jemalloc and tcmalloc should be built from the [General setup](#general-setup) section.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"path_to/tcmalloc/lib/libtcmalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```
* Set IOMP preload for better performance
  ```
  pip install packaging intel-openmp
  export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
  ```

* Set ENV to use AMX if you are using SPR
  ```bash
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```
* Set ENV to use fp16 AMX if you are using a supported platform
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
  ```

## Datasets
The dataset can be downloaded and preprocessed by following https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm#create-the-synthetic-multi-hot-dataset.
We also provided a preprocessed scripts based on the instruction above. `preprocess_raw_dataset.sh`.
After you loading the raw dataset `day_*.gz` and unzip them to RAW_DIR.
```bash
cd <AI Reference Models>/models_v2/pytorch/torchrec_dlrm/inference/cpu
export MODEL_DIR=$(pwd)
export RAW_DIR=<the unziped raw dataset>
export TEMP_DIR=<where your choose the put the temp file during preprocess>
export PREPROCESSED_DIR=<where your choose the put the one-hot dataset>
export MULTI_HOT_DIR=<where your choose the put the multi-hot dataset>
bash preprocess_raw_dataset.sh
```

## Pre-Trained checkpoint
You can download and unzip checkpoint by following
https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch#downloading-model-weights

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/torchrec_dlrm/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Install general model requirements
    ```
    ./setup.sh
    ```
5. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation).

6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY)              | `export TEST_MODE=THROUGHPUT`                  |
| **DATASET_DIR**             |                               `export DATASET_DIR=<multi-hot dataset dir>`                                  |
| **WEIGHT_DIR** (ONLY FOR ACCURACY)     |                 `export WEIGHT_DIR=<offical released checkpoint>`        |
| **PRECISION**    |                               `export PRECISION=int8 <specify the precision to run: int8, fp32, bf32 or bf16>`                             |
| **OUTPUT_DIR**    |                               `export OUTPUT_DIR=$PWD`                               |
| **BATCH_SIZE** (optional)  |                               `export BATCH_SIZE=10000`                                |
7. Run `run_model.sh`
## Output

Single-tile output will typically look like:

```
accuracy 76.215 %, best 76.215 %
dlrm_inf latency:  0.11193203926086426  s
dlrm_inf avg time:  0.007462135950724284  s, ant the time count is : 15
dlrm_inf throughput:  4391235.996821996  samples/s
```


Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 4391236.0
   unit: inst/s
 - key: latency
   value: 0.007462135950724283
   unit: s
 - key: accuracy
   value: 76.215
   unit: accuracy
```
