<!--- 0. Title -->
# BERT Large training for Intel® Data Center GPU Max Series

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large training using
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
  - Intel® oneAPI Collective Communications Library (oneCCL), required by Intel® Optimization for Horovod* only
  * Download and install the verified DPC++ compiler, oneTBB and oneMKL.
    
    ```bash
    $ wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19079/l_BaseKit_p_2023.0.0.25537_offline.sh
    # 4 components are necessary: DPC++/C++ Compiler, DPC++ Libiary, oneTBB and oneMKL
    # if you want to run distributed training with Intel® Optimization for Horovod*, oneCCL is needed too(Intel® oneAPI MPI Library will be installed automatically as its dependency)
    $ sudo sh ./l_BaseKit_p_2023.0.0.25537_offline.sh
    ```
    For any more details on instructions on how to download and install the base-kit, please follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=offline.

  - Set environment variables
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh

    # oneCCL (and Intel® oneAPI MPI Library as its dependency), required by Intel® Optimization for Horovod* only
    source {ONEAPI_ROOT}/mpi/latest/env/vars.sh
    source {ONEAPI_ROOT}/ccl/latest/env/vars.sh
    ```

<!--- 30. Datasets -->
## Datasets

### Pretrained models

Download and extract the bert large uncased (whole word masking) pretrained model checkpoints
from the [google bert repo](https://github.com/google-research/bert#pre-trained-models).
The extracted directory should be set to the `BERT_LARGE_DIR` environment
variable when running the quickstart scripts. A dummy dataset will be auto generated and 
used for training scripts.


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`bfloat16_training.sh`](bfloat16_training.sh) | bfloat16 precision script for bert large pretraining |
| [`bfloat16_training_hvd.sh`](bfloat16_training_hvd.sh) | bfloat16 precision script for bert large pretraining with Intel® Optimization for Horovod* support |


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
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading the pretrained model. A path to
this directory will need to be set in the `BERT_LARGE_DIR`
environment variable prior to running a [quickstart script](#quick-start-scripts).

### Run the model on Baremetal
Navigate to the BERT Large training directory, and set environment variables:
```
cd models

export OUTPUT_DIR=<path where output log files will be written>
export PRECISION=bfloat16
export BERT_LARGE_DIR=<path to the wwm_uncased_L-24_H-1024_A-16 directory>

# Set the following `Tile` env variable only for running `bfloat16_training.sh` script:
export Tile=2

# Run `bfloat16_training.sh` script:
./quickstart/language_modeling/tensorflow/bert_large/training/gpu/bfloat_training.sh

# To run `bfloat16_training_hvd.sh` script:
# Install `bfloat16_training_hvd.sh` script specific dependencies:
./quickstart/language_modeling/tensorflow/bert_large/training/gpu/setup.sh
./quickstart/language_modeling/tensorflow/bert_large/training/gpu/bfloat_training_hvd.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

