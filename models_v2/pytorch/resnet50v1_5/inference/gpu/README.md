# ResNet50v1.5 Inference

ResNet50v1.5 Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    Pytorch    |       -        |           -           |         -          |

# Pre-Requisite
* Host has one of the following GPUs:
  * **Arc Series** - [Intel® Arc™ A-Series Graphics](https://ark.intel.com/content/www/us/en/ark/products/series/227957/intel-arc-a-series-graphics.html)
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
## Dataset: imagenet
Default is dummy dataset.
ImageNet is recommended, the download link is https://image-net.org/challenges/LSVRC/2012/2012-downloads.php.

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/resnet50v1_5/inference/gpu`
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
|:----------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=False` (provide True for multi-tile GPU such as Max 1550, and False for single-tile GPU such as Max 1100 or Flex Series GPU or Arc Series GPU)         |
| **PLATFORM**                 | `export PLATFORM=Max` (Max or Flex or Arc)                                           |
| **NUM_DEVICES**              | `export NUM_DEVICES=<num_devices>` (`<num_devices>` is the number of GPU devices used for inference. If it is larger than 1, the script launches multi-instance inference, where 1 instance launched on each GPU device simultaneously. It must be equal to or smaller than the number of GPU devices attached to each node. For GPU with 2 tiles, such as Max 1550 GPU, the number of GPU devices in each node is 2 times the number of GPUs, so `<num_devices>` can be set as <=16 for a node with 8 Max 1550 GPUs. While for GPU with single tile, such as Max 1100 GPU or Flex Series GPU or Arc Series GPU, the number of GPU devices available in each node is the same as number of GPUs, so `<num_devices>` can be set as <=8 for a node with 8 single-tile GPUs.)                             |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=/the/path/to/output_dir`            |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=1024`                               |
| **PRECISION** (optional)     |`export PRECISION=INT8` (INT8,FP32, FP16 for all platform, BF16 and TF32 only for Max)|
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=500`                            |
| **DATASET_DIR** (optional)   |                               `export DATASET_DIR=--dummy` (provide --dummy if using dummy dataset or </the/path/to/dataset> if using Imagenet)                      |
8. Run `run_model.sh`

## Output

Single-device output will typically look like:

```
Test: [500/500] Time  0.039 ( 0.042)    Loss 8.4575e+00 (8.4625e+00)    Acc@1   0.20 (  0.10)   Acc@5   0.59 (  0.50)
Quantization Evalution performance: batch size:1024, throughput:26373.51 image/sec, Acc@1:0.10, Acc@5:0.50
```

Multi-device output will typically look like:
```
[1]     Test: [500/500] Time  0.040 ( 0.044)    Loss 8.4575e+00 (8.4625e+00)    Acc@1   0.20 (  0.10)   Acc@5   0.59 (  0.50)
Quantization Evalution performance: batch size:1024, throughput:25780.13 image/sec, Acc@1:0.10, Acc@5:0.50
[2]     Test: [500/500] Time  0.039 ( 0.044)    Loss 8.4575e+00 (8.4625e+00)    Acc@1   0.20 (  0.10)   Acc@5   0.59 (  0.50)
Quantization Evalution performance: batch size:1024, throughput:26216.49 image/sec, Acc@1:0.10, Acc@5:0.50
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 26373.51
   unit: fps
 - key: latency
   value: 0.0388268379900893
   unit: s
 - key: accuracy
   value: 0.100
   unit: Acc@1
```
