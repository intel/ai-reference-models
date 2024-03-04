# RNNT Training

RNNT Training best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Training   |    Pytorch    |       https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch        |           -           |         -          |

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
##Get Dataset
Downloading and preprocessing the fully dataset requires 500GB of free disk space and can take several hours to complete.

This repository provides scripts to download, and extract the following datasets:

```bash
bash scripts/download_librispeech.sh
```
Once the data download is complete, the following folders should exist:

* `datasets/LibriSpeech/`
   * `train-clean-100/`
   * `dev-clean/`
   * `dev-other/`
   * `test-clean/`
   * `test-other/`

Next, convert the data into WAV files:
```bash
bash scripts/preprocess_librispeech.sh
```
Once the data is converted, the following additional files and folders should exist:
* `datasets/LibriSpeech/`
   * `librispeech-train-clean-100-wav.json`
   * `librispeech-train-clean-360-wav.json`
   * `librispeech-train-other-500-wav.json`
   * `librispeech-dev-clean-wav.json`
   * `librispeech-dev-other-wav.json`
   * `librispeech-test-clean-wav.json`
   * `librispeech-test-other-wav.json`
   * `train-clean-100-wav/`
   * `train-clean-360-wav/`
   * `train-other-500-wav/`
   * `dev-clean-wav/`
   * `dev-other-wav/`
   * `test-clean-wav/`
   * `test-other-wav/`

For training, the following manifest files are used:
   * `librispeech-train-clean-100-wav.json`
   * `librispeech-train-clean-360-wav.json`
   * `librispeech-train-other-500-wav.json`
please specify the folder path which contains datasets/LibriSpeech as parameter DATASET_DIR


## Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/rnnt/training/gpu`
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
7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=True` (True or False)                                             |
| **PLATFORM**                 | `export PLATFORM=Max` (Max)                                                 |
| **DATASET_DIR**              |                               `export DATASET_DIR=`                                  |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=512`                                |
| **PRECISION** (optional)     |                               `export PRECISION=BF16` (BF16 FP32 and TF32 for Max)   |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
8. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
Done in 172.56718969345093
total samples tested:  10240
total time (excluded audio processing):  88.60093999199307 s
rnnt_train throughput: 115.574 sample per second
rnnt_train latency:  4.430046999599654
Saving module RNNT in /home/gta/wliao2/frameworks.ai.models.intel-models/models/language_modeling/pytorch/rnnt/training/gpu/RNNT_1703175379.689953-epoch-0.pt
Saved.
```

Multi-tile output will typically looks like:
```
Done in 395.0894150733948
[0] total samples tested:  [0] 10240[0]
[0] total time (excluded audio processing):  [0] 89.35191706901242 [0] s
[0] rnnt_train throughput: 114.603 sample per second
[0] rnnt_train latency: [0]  [0] 4.467595853450621
[0] Saving module DistributedDataParallel in /home/gta/wliao2/frameworks.ai.models.intel-models/models/language_modeling/pytorch/rnnt/training/gpu/out/DistributedDataParallel_1703176458.9948287-epoch-0.pt
[0] Saved.
[1] sample id  22   | cost time is [1]  4.505866523002624
 79%|███████▊  | 22/28 [06:41<01:49, 18.25s/it]
[1] total samples tested:  10240
[1] total time (excluded audio processing):  91.75176179101254 [1] s
[1] rnnt_train throughput: 111.605 sample per second[1]
[1] rnnt_train latency:  [1] 4.587588089550628
[1] Saving module DistributedDataParallel in /home/gta/wliao2/frameworks.ai.models.intel-models/models/language_modeling/pytorch/rnnt/training/gpu/out/DistributedDataParallel_1703176465.345023-epoch-0.pt
[1] Saved.
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 226.208
   unit: fps
 - key: latency
   value: N/A
   unit: s
 - key: accuracy
   value: None
   unit: loss
```
