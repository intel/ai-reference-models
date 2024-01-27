# DLRM v1 Inference

DLRM v1 Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/facebookresearch/dlrm        |           -           |         -          |

# Pre-Requisite
* Host has [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html)
* Host has installed latest Intel® Data Center GPU Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* Host has installed latest Intel® Data Center GPU Max & Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.

# Prepare Dataset
The code supports interface with the [Criteo Kaggle Display Advertising Challenge Dataset](https://ailab.criteo.com/ressources/).
   - Please do the following to prepare the dataset for use with DLRM code:
     - First, specify the raw data file (train.txt) as downloaded with
     - This is then pre-processed (categorize, concat across days...) to allow using with dlrm code
     - The processed data is stored as *.npz file
datset dir need have train.txt and kaggleAdDisplayChallenge_processed.npz

you can get the checkpoints by running the command
./bench/dlrm_s_criteo_kaggle.sh [--test-freq=1024]


## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/dlrm/inference/gpu`
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
| **MULTI_TILE**               | `export MULTI_TILE=False` (False)                                             |
| **PLATFORM**                 | `export PLATFORM=Flex` (Flex)                                                 |
| **DATASET_DIR**              |                               `export DATASET_DIR=`                                  |
| **CKPT_DIR**                 |                               `export CKPT_DIR=`                                     |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=32768`                                |
| **PRECISION** (optional)     |                               `export PRECISION=fp16` (fp16 and fp32 for Flex)                               |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=20`                             |
8. Run `run_model.sh`

> [!NOTE]
> Refer to [CONTAINER.md](CONTAINER.md) for DLRM-v1 inference instructions using docker containers.
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
