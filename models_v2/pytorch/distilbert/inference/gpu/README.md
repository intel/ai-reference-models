# DistilBert Inference

DistilBert Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/huggingface/transformers/tree/main/src/transformers/models/distilbert        |           -           |         -          |

# Pre-Requisite
* Host has one of the following GPUs:
  * **Flex Series** - [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html)
  * **Max Series** - [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)
* Host has installed latest Intel® Data Center GPU Max & Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.

# Prepare Dataset
## Dataset: 
Please refer to https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch-22.09#download-and-prepare-the-data

the dataset should be like below
|_hdf5  
      |_ eval                               # evaluation chunks in binary hdf5 format fixed length (not used in training, can delete after data   preparation)  
      |_ eval_varlength                     # evaluation chunks in binary hdf5 format variable length *used for training*
      |_ training                           # 500 chunks in binary hdf5 format 
      |_ training_4320                      # 
      |_ hdf5_4320_shards_uncompressed   # sharded data in hdf5 format fixed length (not used in training, can delete after data   preparation)
      |_ hdf5_4320_shards_varlength      # sharded data in hdf5 format variable length *used for training

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/distilbert/inference/gpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install the latest GPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation):
  ```
  python -m pip install torch==<torch_version> torchvision==<torchvvision_version> intel-extension-for-pytorch==<ipex_version> --extra-index-url https://pytorch-extension.intel.com/release-whl-aitools/
  ```
6. Set environment variables for Intel® oneAPI Base Toolkit: 
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh
    source {ONEAPI_ROOT}/mpi/latest/env/vars.sh
    source {ONEAPI_ROOT}/ccl/latest/env/vars.sh
    ```
7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=True` (True or False)                                             |
| **PLATFORM**                 | `export PLATFORM=Max` (Max or Flex)                                                 |
| **DATASET_DIR**              |                               `export DATASET_DIR=`                                  |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=32`                                |
| **PRECISION** (optional)     | `export PRECISION=BF16` (FP32, BF16, FP16 and TF32 for Max and FP16, FP32 for Flex)|
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=300`                             |
8. Run `run_model.sh`

> [!NOTE]
> Refer to [CONTAINER_FLEX.md](CONTAINER_FLEX.md) and [CONTAINER_MAX.md](CONTAINER_MAX.md) for DistilBERT inference instructions using docker containers.
## Output

Single-tile output will typically looks like:

```
12/21/2023 14:28:08 - INFO - utils - PID: 148054 -  --- Ending inference
12/21/2023 14:28:08 - INFO - utils - PID: 148054 -  Results: {'acc': 0.5852613944871974, 'eval_loss': 1.9857747395833334}
12/21/2023 14:28:08 - INFO - utils - PID: 148054 -  The total_time 8.032912015914917 s, and perf 1115.411196120202 sentences/s for inference
12/21/2023 14:28:08 - INFO - utils - PID: 148054 -  Let's go get some drinks.
```

Multi-tile output will typically looks like:
```
12/21/2023 14:33:13 - INFO - utils - PID: 148381 -  --- Ending inference
12/21/2023 14:33:13 - INFO - utils - PID: 148381 -  Results: {'acc': 0.5852613944871974, 'eval_loss': 1.9857747395833334}
12/21/2023 14:33:13 - INFO - utils - PID: 148381 -  The total_time 8.122166156768799 s, and perf 1103.1539895958633 sentences/s for inference
-Iter:   5%|▍         | 296/6087 [00:12<03:27, 27.93it/s]12/21/2023 14:33:13 - INFO - utils - PID: 148381 -  Let's go get some drinks.
-Iter:   5%|▍         | 300/6087 [00:12<03:56, 24.42it/s]
12/21/2023 14:33:13 - INFO - utils - PID: 148383 -  --- Ending inference
12/21/2023 14:33:13 - INFO - utils - PID: 148383 -  Results: {'acc': 0.5852613944871974, 'eval_loss': 1.9857747395833334}
12/21/2023 14:33:13 - INFO - utils - PID: 148383 -  The total_time 8.266947984695435 s, and perf 1083.834084427241 sentences/s for inference
12/21/2023 14:33:13 - INFO - utils - PID: 148383 -  Let's go get some drinks.
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 2186.9881
   unit: sent/s
 - key: latency
   value: 0.0292663
   unit: s
 - key: accuracy
   value: 0.5850
   unit: acc
```
