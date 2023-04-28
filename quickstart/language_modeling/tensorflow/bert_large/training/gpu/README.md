<!--- 0. Title -->
# BERT Large training for Intel® Data Center GPU Max Series

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large training using
Intel-optimized TensorFlow with Intel® Data Center GPU Max Series.

<!--- 20. GPU Setup -->
## Software Requirements:
- Intel® Data Center GPU Max Series
- Follow [instructions](https://intel.github.io/intel-extension-for-tensorflow/latest/get_started.html) to install the latest ITEX version and other prerequisites.

- Intel® oneAPI Base Toolkit: Need to install components of Intel® oneAPI Base Toolkit
  - Intel® oneAPI DPC++ Compiler
  - Intel® oneAPI Threading Building Blocks (oneTBB)
  - Intel® oneAPI Math Kernel Library (oneMKL)
  - Follow [instructions](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=offline) to download and install the latest oneAPI Base Toolkit.

  - Set environment variables for Intel® oneAPI Base Toolkit: 
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh
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

