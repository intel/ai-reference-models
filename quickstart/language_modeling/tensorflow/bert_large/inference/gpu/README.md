<!--- 0. Title -->
# BERT Large inference

<!-- 10. Description -->

This document has instructions for running BERT Large inference using
Intel-optimized TensorFlow with Intel® Data Center GPU Max Series.

<!--- 20. GPU Setup -->
## Hardware Requirements:
- Intel® Data Center GPU Max Series, Driver Version: [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html)

## Software Requirements:
- Intel® Data Center GPU Max Series
- Intel GPU Drivers: Intel® Data Center GPU Max Series [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html)
- Intel® oneAPI Base Toolkit 2023.0
- TensorFlow 2.11.0 or 2.10.0
- Python 3.7-3.10
- pip 19.0 or later (requires manylinux2014 support)

  |Release|Intel GPU|Install Intel GPU Driver|
    |-|-|-|
    |v1.1.0|Intel® Data Center GPU Max Series|  Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/index.html#intel-data-center-gpu-max-series) for latest driver installation. If install the verified Intel® Data Center GPU Max Series/Intel® Data Center GPU Flex Series [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html), please append the specific version after components.|

- Intel® oneAPI Base Toolkit 2023.0.0: Need to install components of Intel® oneAPI Base Toolkit
  - Intel® oneAPI DPC++ Compiler
  - Intel® oneAPI Threading Building Blocks (oneTBB)
  - Intel® oneAPI Math Kernel Library (oneMKL)
  * Download and install the verified DPC++ compiler, oneTBB and oneMKL.
    
    ```bash
    $ wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19079/l_BaseKit_p_2023.0.0.25537_offline.sh
    # 4 components are necessary: DPC++/C++ Compiler, DPC++ Libiary, oneTBB and oneMKL
    $ sudo sh ./l_BaseKit_p_2023.0.0.25537_offline.sh
    ```
    For any more details on instructions on how to download and install the base-kit, please follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=offline.

  - Set environment variables
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh
    ```

<!--- 30. Datasets -->
## Datasets

### BERT Large Data
Download and unzip the BERT Large uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Then, download the Stanford Question Answering Dataset (SQuAD) dataset file `dev-v1.1.json` into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

```
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```
Set the `DATASET_DIR` to point to that directory when running BERT Large inference using the SQuAD data.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`benchmark.sh`](benchmark.sh) | This script runs bert large fp16 and fp32 inference. |
| [`accuracy.sh`](accuracy.sh) | This script runs bert large fp16 and fp32 inference in accuracy mode. |


<!--- 50. Baremetal -->
## Run the model
Install the following pre-requisites:
* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
* Install TensorFlow and Intel® Extension for TensorFlow (ITEX):

  The Intel® Extension for TensorFlow* requires stock TensorFlow, and the version should be == 2.11.0 or 2.10.0.

  On Linux, it is often necessary to first update pip to a version that supports manylinux2014 wheels.
  ```bash
  pip install --upgrade pip
  ```
  
  ```bash
  pip install tensorflow==2.11.0
  pip install --upgrade intel-extension-for-tensorflow[gpu]
  ```
   To verify that TensorFlow and ITEX are correctly installed:
  ```
  python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
  ```
  
* Download the frozen graph model file, and set the FROZEN_GRAPH environment variable to point to where it was saved:
  ```bash
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/fp32_bert_squad.pb
  ```

* Download the pretrained model directory and set the PRETRAINED_DIR environment variable to point where it was saved:
  ```bash
  wget  https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
  unzip wwm_uncased_L-24_H-1024_A-16.zip
  ```

* Download the SQUAD directory and set the SQUAD_DIR environment variable to point where it was saved:
  ```bash
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
  wget https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
  ```
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```

### Run the model on Baremetal
Navigate to the BERT Large inference directory, and set environment variables:
```
cd models
export OUTPUT_DIR=<path where output log files will be written>
export PRECISION=<Set precision: fp16 or fp32>
export FROZEN_GRAPH=<path to pretrained model file (*.pb)>
export PRETRAINED_DIR=<path to pretrained directory>
export SQUAD_DIR=<path to squad directory>

# Set `Tile` env variable only for running `benchmark.sh` script:
export Tile=2

# Run quickstart script:
./quickstart/language_modeling/tensorflow/bert_large/inference/gpu/<script name>.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

