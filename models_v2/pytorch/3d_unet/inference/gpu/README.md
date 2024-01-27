# 3D-UNet Inference

3D-UNet Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    Pytorch    |       https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-brats19        |           -           |         -          |

# Pre-Requisite
* Host has Intel® Data Center GPU Max Series - [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)
* Host has installed latest Intel® Data Center GPU Max Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.


# Prepare Dataset
* Please download [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) separately and unzip the dataset.

## Prepare data file
* Download the data file from https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-brats19/folds, put them under the folder models_v2/pytorch/3d_unet/inference/gpu/folds 

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/3d_unet/inference/gpu`
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
    python -m pip install torch==<torch_version> torchvision==<torchvision_version> intel-extension-for-pytorch==<ipex_version> --extra-index-url https://pytorch-extension.intel.com/release-whl-aitools/
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
5. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=True` (True or False)                                             |
| **PLATFORM**                 | `export PLATFORM=Max` (Max)                                                 |
| **DATASET_DIR**              |                               `export DATASET_DIR=`                                  |
| **PRECISION** (optional)     |                               `export PRECISION=FP16` (FP16, INT8 and FP32 for Max)                               |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
6. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
3dunet_inf throughput:  12.63259639639566  sample/s
3dunet_inf latency:  0.07916029045979182  s
Done!
Destroying SUT...
Destroying QSL...
```

Multi-tile output will typically looks like:
```
3dunet_inf throughput:  12.63259639639566  sample/s
3dunet_inf latency:  0.07916029045979182  s
Done!
Destroying SUT...
Destroying QSL...
3dunet_inf throughput:  12.638104577393317  sample/s
3dunet_inf latency:  0.07912578930457433  s
Done!
Destroying SUT...
Destroying QSL...
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 25.2707
   unit: sample/s
 - key: latency
   value: 0.079143
   unit: s
 - key: accuracy
   value: None
   unit: mean
```
