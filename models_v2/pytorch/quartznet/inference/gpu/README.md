# Quartznet Inference

Quartznet Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    Pytorch    |       https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/QuartzNet        |           -           |         -          |

# Pre-Requisite
* Host has [Intel® Data Center GPU Flex Series](https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html)
* Host has installed latest Intel® Data Center GPU Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.

# Prepare Dataset
This repository provides scripts to download and extract LibriSpeech [http://www.openslr.org/12](http://www.openslr.org/12). The dataset contains 1000 hours of 16kHz read English speech derived from public domain audiobooks from the LibriVox project and has been carefully segmented and aligned. For more information, see the [LIBRISPEECH: AN ASR CORPUS BASED ON PUBLIC DOMAIN AUDIO BOOKS](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) paper.

Set the `DATASET_DIR` environment variable to point to the directory of the dataset

   download and extract the datasets into the required format for later training and inference:
   ```bash
   bash scripts/download_librispeech.sh
   ```
   After the data download is complete, the following folders should exist:
   ```bash
   datasets/LibriSpeech/
   ├── dev-clean
   ├── dev-other
   ├── test-clean
   ├── test-other
   ├── train-clean-100
   ├── train-clean-360
   └── train-other-500
   ```

   Next, convert the data into WAV files:
   ```bash
   bash scripts/preprocess_librispeech.sh
   ```

   After the data is converted, the following additional files and folders should exist:
   ```bash
   datasets/LibriSpeech/
   ├── dev-clean-wav
   ├── dev-other-wav
   ├── librispeech-train-clean-100-wav.json
   ├── librispeech-train-clean-360-wav.json
   ├── librispeech-train-other-500-wav.json
   ├── librispeech-dev-clean-wav.json
   ├── librispeech-dev-other-wav.json
   ├── librispeech-test-clean-wav.json
   ├── librispeech-test-other-wav.json
   ├── test-clean-wav
   ├── test-other-wav
   ├── train-clean-100-wav
   ├── train-clean-360-wav
   └── train-other-500-wav
```

# Download checkpoint
During inference, QuartzNet models trained with [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) can also be used, for instance one of pre-trained models. Set the `CKPT_DIR` environment variable to point to the directory of the `pretrained_model/quartznet_en`.

To download automatically, run:
```bash
# Running following will create a folder called `pretrained_models`. Point the path to `pretrained_models/quartznet_en` directory and set `CKPT_DIR` environment variable
bash scripts/download_quartznet.sh

# For Catalan, French, German, Italian, Mandarin Chinese, Polish, Russian or Spanish available on [NGC](https://ngc.nvidia.com/).
bash scripts/download_quartznet.sh [ca|fr|de|it|zh|pl|ru|es]
```

Pre-trained models can be explicitly converted from the `.nemo` checkpoint format to `.pt` and vice versa.
For more details, run:
```bash
python nemo_dle_model_converter.py --help
``` 

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/quartznet/inference/gpu`
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
| **OUTPUT_DIR**               |  `export OUTPUT_DIR=$PWD`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=8`                                |
| **PRECISION** (optional)     |                               `export PRECISION=fp16`  (fp16 for Flex)                              |
8. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
---latency=0.45642547607421874 s
---throughput=17.527505407474518 fps
WER=5.823529411764706 %
```


Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 17.527505
   unit: fps
 - key: latency
   value: 0.45642547607421874
   unit: s
 - key: accuracy
   value: 5.824
   unit: WER
```
