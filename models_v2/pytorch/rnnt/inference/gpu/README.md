# RNNT Inference

RNNT Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    Pytorch    |       https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch        |           -           |         -          |

# Pre-Requisite
* Host has Intel® Data Center GPU Max - [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)
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
For inference, the `librispeech-dev-clean-wav.json` is used.
please specify the folder path which contains datasets/LibriSpeech as parameter `DATASET_DIR`

```bash
# checkpoint
wget -O rnnt_ckpt.pt https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1
```
Specify the folder which contains the `rnnt_ckpt.pt` model as parameter `WEIGHT_DIR`

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/rnnt/inference/gpu`
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
| **WEIGHT_DIR**              |                               `export WEIGHT_DIR=`                                  |
| **OUTPUT_DIR**              |                               `export OUTPUT_DIR=$PWD`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=512`                                |
| **PRECISION** (optional)     |                               `export PRECISION=BF16` (BF16 FP16 and FP32 for Max )                                |
8. Run `run_model.sh`

## Output

Single-tile output will typically look like:

```
========================>>>>>>>>>>>>>>>>>>>>>

Evaluation WER: 0.07343579777993699

Accuracy: 0.926564202220063

total samples tested:  15360
total time (encoder + decoder, excluded audio processing):  89.80226326499542 s
dataset size:  21624
rnnt_inf throughput: 171.042 sample per second
rnnt_inf latency:  0.005846501514648139
```

Multi-tile output will typically look like:
```
========================>>>>>>>>>>>>>>>>>>>>>

Evaluation WER: 0.07343579777993699

Accuracy: 0.926564202220063

total samples tested:  15360
total time (encoder + decoder, excluded audio processing):  90.02906056994107 s
dataset size:  21624
rnnt_inf throughput: 170.612 sample per second
rnnt_inf latency:  0.0058612669641888715
get in multi_gpu?  False

========================>>>>>>>>>>>>>>>>>>>>>
Evaluation WER: 0.07343579777993699

Accuracy: 0.926564202220063

total samples tested:  15360
total time (encoder + decoder, excluded audio processing):  94.49991247599974 s
dataset size:  21624
rnnt_inf throughput: 162.540 sample per second
rnnt_inf latency:  0.0061523380518228995
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 171.042
   unit: fps
 - key: latency
   value: 2.9934168215993733
   unit: s
 - key: accuracy
   value: 0.927
   unit: accuracy
```
