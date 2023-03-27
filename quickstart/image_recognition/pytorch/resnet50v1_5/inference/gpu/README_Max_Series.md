<!--- 0. Title -->
# ResNet50v1.5 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50v1.5 inference using
Intel(R) Extension for PyTorch with GPU.

<!--- 20. GPU Setup -->
## Hardware Requirements:
- Intel® Data Center GPU Max Series, Driver Version: [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html)

## Software Requirements:
- Intel® Data Center GPU Max Series
- Intel GPU Drivers: Intel® Data Center GPU Max Series [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html)
- Intel® oneAPI Base Toolkit 2023.0
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

The [ImageNet](http://www.image-net.org/) validation dataset is used.

Download and extract the ImageNet2012 dataset from http://www.image-net.org/,
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

A after running the data prep script, your folder structure should look something like this:

```
imagenet
└── val
    ├── ILSVRC2012_img_val.tar
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   ├── ILSVRC2012_val_00006697.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` directory should be set as the
`DATASET_DIR`
(for example: `export DATASET_DIR=/home/<user>/imagenet`).

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| inference_block_format.sh | Runs ResNet50 inference (block format) for int8 precision |

<!--- 50. Baremetal -->
## Run the model
Install the following pre-requisites:
* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
* Install PyTorch and Intel® Extension for PyTorch for GPU (IPEX):
  ```bash
  python -m pip install torch==1.13.0a0 -f https://developer.intel.com/ipex-whl-stable-xpu
  python -m pip install intel_extension_for_pytorch==1.13.10+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
  ```
  To verify that PyTorch and IPEX are correctly installed:
  ```bash
  python -c "import torch;print(torch.device('xpu'))"  # Sample output: "xpu"
  python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.is_available())"  #Sample output True
  python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.has_onemkl())"  # Sample output: True
  ```
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```
* Navigate to ResNet50v1.5 inference directory and install model specific dependencies for the workload:
  ```bash
  # Navigate to the model zoo repo
  cd models

  cd quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu
  ./setup.sh
  cd -
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading and preprocessing the ImageNet dataset. The path to the ImageNet
dataset files will need to be set as the `DATASET_DIR` environment variable
prior to running a [quickstart script](#quick-start-scripts).

### Run the model on Baremetal
Set environment variables for the path to your dataset, an output directory to run the quickstart script:
```
# To run with ImageNet data, the dataset directory will need to be specified in addition to an output directory and precision.
export DATASET_DIR=<path to the preprocessed imagenet dataset>
export OUTPUT_DIR=<Path to save the output logs>
export Tile=2

# Optional envs
export BATCH_SIZE=<Set batch_size else it will run with default batch>

# Run a quickstart script
./quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/inference_block_format.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
