# ResNet50v1.5 Training

ResNet50v1.5 Training best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Training    |    Pytorch    |       -        |           -           |         -          |

# Pre-Requisite
* Host has one of the following GPUs:
  * **Arc Series** - [Intel® Arc™ A-Series Graphics](https://ark.intel.com/content/www/us/en/ark/products/series/227957/intel-arc-a-series-graphics.html)
  * **Max Series** - [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)
* Host has installed latest Intel® Data Center GPU Max & Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library
  - Intel® oneAPI CCL Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.

# Prepare Dataset
## Dataset: imagenet
ImageNet is recommended, the download link is https://image-net.org/challenges/LSVRC/2012/2012-downloads.php.

## Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/resnet50v1_5/training/gpu`
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
7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=False` (True or False)                                            |
| **PLATFORM**                 | `export PLATFORM=Max` (Max or ARC)                                                 |
| **DATASET_DIR**              |                               `export DATASET_DIR=`                                  |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256`                                |
| **PRECISION**  (optional)    | `export PRECISION=BF16` (BF16 or TF32 or FP32 for Max and BF16 or FP32 for ARC )    |
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=20`                             |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
6. Run `run_model.sh`

## Output

Single-tile output will typically look like:

```
Epoch: [0][ 20/196]     Time  0.148 ( 0.379)    Data  0.000 ( 0.218)    Loss 7.3804e+00 (8.0065e+00)    Acc@1   0.00 (  0.12)    Acc@5   0.39 (  0.45)
Training performance: batch size:256, throughput:1716.10 image/sec
```

Multi-tile output will typically looks like:
```
[1] Epoch: [0][  20/2503]       Time  0.173 ( 0.855)    Data  0.001 ( 0.072)    Loss 7.6168e+00 (7.5532e+00)     Acc@1   0.39 (  0.08)   Acc@5   0.39 (  0.43)
[1] Training performance: batch size:256, throughput:1474.26 image/sec
[0] Epoch: [0][  20/2503]       Time  0.172 ( 0.861)    Data  0.001 ( 0.065)    Loss 7.6672e+00 (7.5610e+00)     Acc@1   0.00 (  0.16)   Acc@5   0.00 (  0.74)
[0] Training performance: batch size:256, throughput:1473.24 image/sec
```
Final results of the training run can be found in `results.yaml` file.

```
results:
 - key: throughput
   value: 1716.1
   unit: image/s
 - key: latency
   value: 0.14917545597575899
   unit: s
 - key: accuracy
   value: 7.380
   unit: loss
```
