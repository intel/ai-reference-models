# FastPitch Inference

FastPitch Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch        |           -           |         -          |

# Pre-Requisite
* Host has [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html)
* Host has installed latest Intel® Data Center GPU Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.

# Prepare Dataset and Pre-trained models
Set the `DATASET_DIR` and `CKPT_DIR` environment variables to point to the directories of the dataset and model respectively.
```bash
# Download dataset
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2

# Pretrained models
# Running following will create a folder called `pretrained_models`. Point the path to `pretrained_models` directory and set `CKPT_DIR` environment variable
bash scripts/download_models.sh fastpitch
bash scripts/download_models.sh hifigan

# Prepare mel files from groundtruth
bash scripts/prepare_dataset.sh
```

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/fastpitch/inference/gpu`
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
| **PLATFORM**                 | `export PLATFORM=Flex` (Flex)                                                 |
| **DATASET_DIR**              | `export DATASET_DIR=`                                                                  |
| **CKPT_DIR**                 | `export CKPT_DIR=`                                                                     |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=$PWD`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=8`                                |
| **NUM_ITERATIONS** (optional)|                               `export NUM_ITERATIONS=100`                               |
8. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
----------------
avg latency:  0.13717432243307842
throughput:  58.31995272951221
DLL 2023-12-28 07:44:57.980371 - () | avg fastpitch tokens/s       18699.91 tokens/s
DLL 2023-12-28 07:44:57.980537 - () | avg fastpitch frames/s      126529.00 frames/s
DLL 2023-12-28 07:44:57.980589 - () | avg fastpitch latency         0.03078 s
DLL 2023-12-28 07:44:57.980625 - () | avg fastpitch RTF              234.38 x
DLL 2023-12-28 07:44:57.980661 - () | avg fastpitch RTF@8           1171.92 x
DLL 2023-12-28 07:44:57.981396 - () | 90% fastpitch latency         0.18967 s
DLL 2023-12-28 07:44:57.981629 - () | 95% fastpitch latency         0.22011 s
DLL 2023-12-28 07:44:57.981832 - () | 99% fastpitch latency         0.27960 s
DLL 2023-12-28 07:44:57.981878 - () | avg_fastpitch_mel-loss :  1.8416595458984375
DLL 2023-12-28 07:44:57.981974 - () | avg hifigan samples/s      5366874.90 samples/s
DLL 2023-12-28 07:44:57.982021 - () | avg hifigan latency           0.14818 s
DLL 2023-12-28 07:44:57.982055 - () | avg hifigan RTF                 48.68 x
DLL 2023-12-28 07:44:57.982087 - () | avg hifigan RTF@8              243.40 x
DLL 2023-12-28 07:44:57.982277 - () | 90% hifigan latency           0.71768 s
DLL 2023-12-28 07:44:57.982467 - () | 95% hifigan latency           0.82678 s
DLL 2023-12-28 07:44:57.982654 - () | 99% hifigan latency           1.04001 s
DLL 2023-12-28 07:44:57.982697 - () | avg_hifigan_mel-loss :  0.1687627613544464
DLL 2023-12-28 07:44:57.982753 - () | avg samples/s              4443916.86 samples/s
DLL 2023-12-28 07:44:57.982793 - () | avg letters/s                 3215.88 letters/s
DLL 2023-12-28 07:44:57.982834 - () | avg latency                   0.17896 s
DLL 2023-12-28 07:44:57.982866 - () | avg RTF                         40.31 x
DLL 2023-12-28 07:44:57.982910 - () | avg RTF@8                      201.54 x
DLL 2023-12-28 07:44:57.983092 - () | 90% latency                   0.90139 s
DLL 2023-12-28 07:44:57.983278 - () | 95% latency                   1.03979 s
DLL 2023-12-28 07:44:57.983472 - () | 99% latency                   1.31028 s
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 58.319953
   unit: fps
 - key: latency
   value: 0.13717432243307842
   unit: s
 - key: accuracy
   value: None
   unit: Acc
```
