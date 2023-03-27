<!--- 0. Title -->
# ResNet50v1.5 training

<!-- 10. Description -->
## Description

This document has instructions to run ResNet50v1.5 training using Intel® Extension for PyTorch for GPU.

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

    # oneCCL (and Intel® oneAPI MPI Library as its dependency), required by Intel® Optimization for Horovod* only
    source {ONEAPI_ROOT}/mpi/latest/env/vars.sh
    source {ONEAPI_ROOT}/ccl/latest/env/vars.sh
    ```

<!--- 30. Datasets -->
## Datasets

Download and extract the ImageNet2012 training and validation dataset from
[http://www.image-net.org/](http://www.image-net.org/),
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

After running the data prep script and extracting the images, your folder structure
should look something like this:
```
imagenet
├── train
│   ├── n02085620
│   │   ├── n02085620_10074.JPEG
│   │   ├── n02085620_10131.JPEG
│   │   ├── n02085620_10621.JPEG
│   │   └── ...
│   └── ...
└── val
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` and `train` directories should be set as the
`DATASET_DIR` environment variable before running the quickstart scripts.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| training_plain_format.sh | Runs ResNet50 training (plain format) using the ImageNet dataset for bf16 using auto channel last one tile or two tile. |
| ddp_training_plain_format_nchw.sh | Runs ResNet50 training (plain format) using the ImageNet dataset for bf16 using NCHW (channel first) with Distributed Deep Learning with Parameter Averaging. |

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

  # To run `ddp_training_plain_format_nchw.sh` oneccl_bind_pt is also needed:
  python -m pip install oneccl_bind_pt==1.13.100+gpu -f https://developer.intel.com/ipex-whl-stable-xpu
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
downloading and extracting the ImageNet dataset.

### Run the model on Baremetal
Set environment variables for the path to your dataset, an output directory to run the quickstart script:
```
# Set environment vars for the dataset and an output directory
export DATASET_DIR=<path the ImageNet directory>
export OUTPUT_DIR=<directory where log files will be written>

# Optional envs
export BATCH_SIZE=<Set batch_size else it will run with default batch>

# Run a quickstart script:
quickstart/image_recognition/pytorch/resnet50v1_5/training/gpu/ddp_training_plain_format_nchw.sh

# Set `Tile` env variable only for `training_plain_format.sh` script
export Tile=2
quickstart/image_recognition/pytorch/resnet50v1_5/training/gpu/training_plain_format.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
