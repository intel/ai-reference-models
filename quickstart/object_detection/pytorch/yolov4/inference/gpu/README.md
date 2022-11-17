<!--- 0. Title -->
# YOLOv4 inference

<!-- 10. Description -->
## Description

This document has instructions for running YOLOv4 inference using
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

Download and extract the 2017 training/validation images and annotations from the
[COCO dataset website](https://cocodataset.org/#download) to a `coco` folder
and unzip the files. After extracting the zip files, your dataset directory
structure should look something like this:
```
coco
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017
│   ├── 000000454854.jpg
│   ├── 000000137045.jpg
│   ├── 000000129582.jpg
│   └── ...
└── val2017
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    ├── 000000000632.jpg
    └── ...
```
The parent of the `annotations`, `train2017`, and `val2017` directory (in this example `coco`)
is the directory that should be used when setting the `image` environment
variable for YOLOv4 (for example: `export image=/home/<user>/coco/val2017/000000581781.jpg`).
In addition, we should also set the `size` environment to match the size of image.
(for example: `export size=416`)

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_with_dummy_data.sh` | Inference with int8 batch_size64 dummy data |

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
* Navigate to Yolov4 inference directory and install model specific dependencies for the workload:
  ```bash
  cd quickstart/object_detection/pytorch/yolov4/inference/gpu
  ./setup.sh
  cd -
  ```
* Download the pretrained weights file, and set the PRETRAINED_MODEL environment variable to point to where it was saved:
  ```bash
  wget https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ
  ```

### Run the model on Baremetal
```
Set environment variables:
export DATASET_DIR=<path where yolov4 COCO dataset>
export PRETRAINED_MODEL=<path to directory where the pretrained weights file was saved>
export OUTPUT_DIR=<Path to save the output logs>

Run the inference script, only int8 precision is supported:
./quickstart/object_detection/pytorch/yolov4/inference/gpu/inference_with_dummy_data.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

