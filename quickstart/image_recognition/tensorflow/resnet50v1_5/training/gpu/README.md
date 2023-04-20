<!--- 0. Title -->
# ResNet50 v1.5 Training

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 v1.5 training using
Intel® Extension for TensorFlow* with Intel® Data Center GPU Max Series.

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

Download and preprocess the ImageNet dataset using the [instructions here](datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`bfloat16_training_full.sh`](bfloat16_training_full.sh) | Runs full bfloat16 training  |
| [`bfloat16_training_hvd.sh`](bfloat16_training_hvd.sh) | Runs bfloat16 training with Intel® Optimization for Horovod* |

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
downloading and preprocessing the ImageNet dataset. The path to the ImageNet
TF records files will need to be set as the `DATASET_DIR` environment variable
prior to running a [quickstart script](#quick-start-scripts).

### Run the model on Baremetal
Navigate to the ResNet50 v1.5 training directory, and set environment variables:
```
cd models
export OUTPUT_DIR=<path where output log files will be written>
export PRECISION=bfloat16
export DATASET_DIR=<path to the preprocessed imagenet dataset directory>

# Optional envs
export BATCH_SIZE=<Set batch_size else it will run with default batch>

# Run quickstart script:
./quickstart/image_recognition/tensorflow/resnet50v1_5/training/gpu/bfloat16_training_hvd.sh

# Set 'Tile' env variable only for running "bfloat16_training_full.sh" script: 
export Tile=2
./quickstart/image_recognition/tensorflow/resnet50v1_5/training/gpu/bfloat16_training_full.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

