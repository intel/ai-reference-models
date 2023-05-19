<!--- 0. Title -->
# YOLOv4 inference

<!-- 10. Description -->
## Description

This document has instructions for running Yolov4 inference using
Intel® Extension for PyTorch with Intel® Data Center GPU Flex Series.

<!--- 20. GPU Setup -->
## Software Requirements:
- Intel® Data Center GPU Flex Series
- Follow [instructions](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html) to install the latest IPEX version and other prerequisites.

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
| `inference_block_format.sh` | Inference with int8 batch_size256 block format |

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

* Navigate to Yolov4 inference directory and install model specific dependencies for the workload:
  ```bash
  cd models
  ./quickstart/object_detection/pytorch/yolov4/inference/gpu/setup.sh
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
./quickstart/object_detection/pytorch/yolov4/inference/gpu/inference_block_format.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

