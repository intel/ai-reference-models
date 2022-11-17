<!--- 0. Title -->
# ResNet50 v1.5 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 v1.5 inference using
Intel(R) Extension for TensorFlow with Intel(R) Data Center GPU Flex Series.

<!--- 20. GPU Setup -->
## Hardware Requirements:
- Intel® Data Center GPU Flex Series

## Software Requirements:
- Ubuntu 20.04 (64-bit)
- Intel GPU Drivers: Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html)

  |Release|OS|Intel GPU|Install Intel GPU Driver|
    |-|-|-|-|
    |v1.0.0|Ubuntu 20.04|Intel® Data Center GPU Flex Series| Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest driver installation. If install the verified Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html), please append the specific version after components, such as `apt-get install intel-opencl-icd=22.28.23726.1+i419~u20.04`|

- Intel® oneAPI Base Toolkit 2022.3: Need to install components of Intel® oneAPI Base Toolkit
  - Intel® oneAPI DPC++ Compiler
  - Intel® oneAPI Math Kernel Library (oneMKL)
  * Download and install the verified DPC++ compiler and oneMKL in Ubuntu 20.04.

    ```bash
    $ wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18852/l_BaseKit_p_2022.3.0.8767_offline.sh
    # 4 components are necessary: DPC++/C++ Compiler, DPC++ Libiary, Threading Building Blocks and oneMKL
    $ sh ./l_BaseKit_p_2022.3.0.8767_offline.sh
    ```
    For any more details, please follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html.

  - Set environment variables
    Default installation location {ONEAPI_ROOT} is /opt/intel/oneapi for root account, ${HOME}/intel/oneapi for other accounts
    ```bash
    source {ONEAPI_ROOT}/setvars.sh
    ```

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| [`online_inference.sh`](online_inference.sh) | Runs online inference for int8 precision | 
| [`batch_inference.sh`](batch_inference.sh)| Runs batch inference for int8 precision |
| [`accuracy.sh`](accuracy.sh) | Measures the model accuracy for int8 precision |

<!--- 50. Baremetal -->
## Run the model
Install the following pre-requisites:
* Python version 3.9
* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
* Install TensorFlow and Intel® Extension for TensorFlow (ITEX):

  Intel® Extension for TensorFlow requires stock TensorFlow v2.10.0 to be installed.
  
  ```bash
  pip install tensorflow==2.10.0
  pip install --upgrade intel-extension-for-tensorflow[gpu]
  ```
   To verify that TensorFlow and ITEX are correctly installed:
  ```
  python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
  ```
* Download the frozen graph model file, and set the FROZEN_GRAPH environment variable to point to where it was saved:
  ```bash
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/gpu/resnet50v1_5_int8_h2d_avg_itex.pb
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading and preprocessing the ImageNet dataset. The path to the ImageNet
TF records files will need to be set as the `DATASET_DIR` environment variable
prior to running a [quickstart script](#quick-start-scripts).

### Run the model on Baremetal
Navigate to the ResNet50 v1.5 inference directory, and set environment variables:
```
export DATASET_DIR=<path to the preprocessed imagenet dataset directory>
export OUTPUT_DIR=<path where output log files will be written>
export PRECISION=int8
export FROZEN_GRAPH=<path to pretrained model file (*.pb)>

Run quickstart script:
./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/<script name>.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

