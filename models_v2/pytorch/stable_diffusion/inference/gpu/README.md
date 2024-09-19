# Stable Diffusion Inference

Stable Diffusion Inference best known configurations with Intel® Extension for PyTorch.

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

# Dataset
The scripts will download the dataset automatically. It uses nateraw/parti-prompts (https://huggingface.co/datasets/nateraw/parti-prompts) as the dataset.

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/stable_diffusion/inference/gpu`
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
| **PLATFORM**                 | `export PLATFORM=Max` (Max or Flex or Arc)                                                 |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=1`                                |
| **PRECISION** (optional)     |           `export PRECISION=fp16` (fp16 and fp32 are supported for all platform)|
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
| **MODEL** (optional)         | `export MODEL='stabilityai/stable-diffusion-2-1'` (must be one of 'stabilityai/stable-diffusion-2-1' (default), "CompVis/stable-diffusion-v1-4" or "stabilityai/stable-diffusion-xl-base-1.0"- note that the xl-base-1.0 will use the StableDiffusionXL pipeline instead)|


8. Run `run_model.sh`

> [!NOTE]
> Refer to [CONTAINER_FLEX.md](CONTAINER_FLEX.md) and [CONTAINER_MAX.md](CONTAINER_MAX.md) for Stable Diffusion Inference instructions using docker containers.
## Output

Single-tile output will typically looks like:

```
No policy available for current head_size 512
inference Latency: 3671.8995094299316 ms
inference Throughput: 0.2723386076966065 samples/s
CLIP score: 33.59451
```

Multi-tile output will typically looks like:
```
26%|██▌       | 13/50 [00:00<00:01, 20.13it/s]inference Latency: 3714.4706646601358 ms
inference Throughput: 0.2692173637320938 samples/s
CLIP score: 33.58945666666667
100%|██████████| 50/50 [00:02<00:00, 19.64it/s]
No policy available for current head_size 512
inference Latency: 3794.5287148157754 ms
inference Throughput: 0.26353733893104825 samples/s
CLIP score: 33.58307666666666
```
please noted that we have using it/s as the throughput. you can find in the `results.yaml`.

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 41.4400
   unit: it/s
it/s
 - key: latency
   value: 0.0482633
   unit: s
 - key: accuracy
   value: 33.5335
   unit: accuracy
```
