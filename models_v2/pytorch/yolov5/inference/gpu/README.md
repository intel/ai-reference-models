# YOLO V5 Inference

YOLO V5 Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       -        |           -           |         -          |

# Pre-Requisite
* Host has [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html)
* Host has installed latest Intel® Data Center GPU Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.


## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models_v2/pytorch/yolov5/inference/gpu`
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
7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=False` (False)                                             |
| **PLATFORM**                 | `export PLATFORM=Flex` (Flex)   
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=$PWD`                                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=32`                                |
| **PRECISION** (optional)     |                               `export PRECISION=fp16`                                |                            |
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=500`                             |
8. Run `run_model.sh`

## Output

Single-tile output will typicall looks like:

```
Latency: 0.05215027490612007
Throughput: 613.6113387245952
```

Multi-tile output will typicall looks like:
```
Latency: 0.05215027490612007
Throughput: 613.6113387245952
```



Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 6093899.0
   unit: inst/s
 - key: latency
   value: 0.00537718137105306
   unit: s
 - key: accuracy
   value: 76.215
   unit: accuracy
```
