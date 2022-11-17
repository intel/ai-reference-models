<!--- 0. Title -->
# SSD-Mobilenetv1 inference

<!-- 10. Description -->
## Description

This document has instructions for running SSD-Mobilenetv1 inference using
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

The [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) validation dataset is used.

Download and extract the VOC2007 dataset from http://host.robots.ox.ac.uk/pascal/VOC/voc2007/,
After extracting the data, your folder structure should look something like this:

```
VOC2007
├── Annotations
│   ├── 000038.xml    
│   ├── 000724.xml
│   ├── 001440.xml
│   └── ...
├── ImageSets
│   ├── Layout    
│   ├── Main
│   └── Segmentation
├── SegmentationClass
│   ├── 005797.png   
│   ├── 007415.png 
│   ├── 006581.png 
│   └── ...
├── SegmentationObject
│   ├── 005797.png    
│   ├── 006581.png
│   ├── 007415.png
│   └── ...
└── JPEGImages
    ├── 002832.jpg    
    ├── 003558.jpg
    ├── 004262.jpg
    └── ...
```
The folder should be set as the `DATASET_DIR`
(for example: `export DATASET_DIR=/home/<user>/VOC2007`).

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_with_dummy_data.sh` | Inference with dummy data, batch size 512, for int8 blocked channel first. |

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
* Navigate to SSD-Mobilenet inference directory and install model specific dependencies for the workload:
  ```bash
  cd quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu
  ./setup.sh
  cd -
  ```
* Download the dataset label file, and set the "label" environment variable to point to where it was saved (for example: `export label=/home/<user>/voc-model-labels.txt`):
  ```bash
  wget https://storage.googleapis.com/models-hao/voc-model-labels.txt
  ```

This snippet shows how to run the inference quickstart script. The inference script
will download the model weights to the directory location set in 'PRETRAINED_MODEL'.

```
### Run the model on Baremetal
Set environment variables:
export DATASET_DIR=<Path to the VOC2007 folder>
export OUTPUT_DIR=<Path to save the output logs>
export PRETRAINED_MODEL=<path to directory where the model weights will be loaded>
export label=<path to label.txt file>

Run the inference script, only int8 precision is supported:
./quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/inference_with_dummy_data.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

