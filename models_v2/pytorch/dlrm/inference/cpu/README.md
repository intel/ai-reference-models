# DLRM v1 Inference

DLRM v1 Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/facebookresearch/dlrm        |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#introduction)
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

* Set ENV to use fp16 AMX if you are using a supported platform
```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```

# Prepare Dataset
### Criteo Terabyte Dataset

The [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) is used to run DLRM. To download the dataset, you will need to visit the Criteo website and accept their terms of use: [https://labs.criteo.com/2013/12/download-terabyte-click-logs/](https://labs.criteo.com/2013/12/download-terabyte-click-logs/). Copy the download URL into the command below as the `<download url>` and replace the `<dir/to/save/dlrm_data>` to any path where you want to download and save the dataset.
```
export DATASET_DIR=<dir/to/save/dlrm_data>
mkdir ${DATASET_DIR} && cd ${DATASET_DIR}
curl -O <download url>/day_{$(seq -s , 0 23)}.gz
gunzip day_*.gz
```
The raw data will be automatically preprocessed and saved as `day_*.npz` to the `DATASET_DIR` when DLRM is run for the first time. On subsequent runs, the scripts will automatically use the preprocessed data.

## Pre-trained Model
Download the DLRM PyTorch weights (`tb00_40M.pt`, 90GB) from the [MLPerf repo](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch#more-information-about-the-model-weights) and set the `WEIGHT_PATH` to point to the weights file.
```
export WEIGHT_PATH=<path to the tb00_40M.pt file>
```

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/dlrm/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Install general model requirements
    ```
    pip install -r requirements.txt
    ```
5. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation).

6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY)              |                               `export TEST_MODE=THROUGHPUT`                                  |
| **DATASET_DIR**              |                               `export DATASET_DIR=<path-to-dlrm_data> or <path-to-preprocessed-data>`                                  |
| **WEIGHT_PATH**      |                 `export WEIGHT_PATH=<path to the tb00_40M.pt file>`        |
| **BATCH_SIZE** (optional)  |                               `export BATCH_SIZE=10000`                                |
| **PRECISION**    |                               `export PRECISION=int8 <specify the precision to run: int8, fp32, bf32 or bf16>`                             |
| **OUTPUT_DIR**    |                               `export OUTPUT_DIR=$PWD`                               |
8. Run `run_model.sh`
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
