<!--- 0. Title -->
# ResNet50v1.5 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50v1.5 inference using
Intel(R) Extension for PyTorch with GPU.

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
    wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18852/l_BaseKit_p_2022.3.0.8767_offline.sh
    # 4 components are necessary: DPC++/C++ Compiler, DPC++ Libiary, Threading Building Blocks and oneMKL
    sh ./l_BaseKit_p_2022.3.0.8767_offline.sh
    ```
    For any more details, please follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html.

  - Set environment variables: 
    Default installation location {ONEAPI_ROOT} is /opt/intel/oneapi for root account, ${HOME}/intel/oneapi for other accounts
    ```bash
    source {ONEAPI_ROOT}/setvars.sh
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
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   ├── ILSVRC2012_val_00006697.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` directory should be set as the
`DATASET_DIR`
(for example: `export DATASET_DIR=/home/<user>/imagenet`).

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| inference_block_format.sh | Runs ResNet50 inference (block format) for the specified precision (int8) |

<!--- 50. Baremetal -->
## Run the model
Install the following pre-requisites:
* Python version 3.9
* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
* Install PyTorch and Intel® Extension for PyTorch for GPU (IPEX):
  ```bash
  python -m pip install torch==1.10.0a0 -f https://developer.intel.com/ipex-whl-stable-xpu
  python -m pip install numpy==1.23.4
  python -m pip install intel_extension_for_pytorch==1.10.200+gpu -f https://developer.intel.com/ipex-whl-stable-xpu
  ```
  To verify that PyTorch and IPEX are correctly installed:
  ```bash
  python -c "import torch;print(torch.device('xpu'))"  # Sample output: "xpu"
  python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.is_available())"  #Sample output True
  python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.has_onemkl())"  # Sample output: True
  ```
* Navigate to ResNet50v1.5 inference directory and install model specific dependencies for the workload:
  ```bash
  cd quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu
  ./setup.sh
  cd -
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading and preprocessing the ImageNet dataset. The path to the ImageNet
dataset files will need to be set as the `DATASET_DIR` environment variable
prior to running a [quickstart script](#quick-start-scripts).

### Run the model on Baremetal
Set environment variables for the path to your dataset, an output directory, and specify the precision to run the quickstart script:
```
To run with ImageNet data, the dataset directory will need to be specified in addition to an output directory and precision.
export DATASET_DIR=<path to the preprocessed imagenet dataset>
export OUTPUT_DIR=<Path to save the output logs>
export PRECISION=int8

# Run a quickstart script
./quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/inference_block_format.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

