<!--- 0. Title -->
# MaskRCNN inference

<!-- 10. Description -->
## Description

This document has instructions for running MaskRCNN inference using
Intel® Extension for TensorFlow with Intel® Data Center GPU Flex Series.

<!--- 20. GPU Setup -->
## Software Requirements:
- Intel® Data Center GPU Flex Series
- Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
- Follow [instructions](https://pypi.org/project/intel-extension-for-tensorflow) to install the latest ITEX version and other prerequisites.

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

This repository provides scripts to download and extract the [COCO 2017 dataset](http://cocodataset.org/#download).

Download and pre-process the datasets using script `download_and_preprocess_coco.sh` provided [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN/dataset). Set the `DATASET_DIR` to point to the TF records directory when running MaskRCNN.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `inference` | Runs inference for FP16 precision on Flex series 170 with default batch size of 16|

<!--- 50. Baremetal -->
## Run the model
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading and preprocessing the dataset. The path to the downloaded dataset will need to be set as the `DATASET_DIR` environment variable prior to running a [quickstart script](#quick-start-scripts).

### Run the model on Baremetal
Navigate to the Model Zoo directory, and set environment variables:
```
cd models
export OUTPUT_DIR=<path where output log files will be written>
export PRECISION=<provide precision,supports fp16>
export DATASET_DIR=<path to the preprocessed COCO dataset>
export BATCH_SIZE=16

Run the model specific dependencies:
NOTE: Installing dependencies in setup.sh may require root privilege
./quickstart/image_segmentation/tensorflow/maskrcnn/inference/gpu/setup.sh

Run quickstart script:
./quickstart/image_segmentation/tensorflow/maskrcnn/inference/gpu/inference.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
