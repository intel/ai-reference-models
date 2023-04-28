<!--- 0. Title -->
# SSD-Mobilenetv1 inference

<!-- 10. Description -->
## Description

This document has instructions for running SSD-MobileNet inference using
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
* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```

* Navigate to SSD-Mobilenet inference directory and install model specific dependencies for the workload:
  ```bash
  cd models
  ./quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/setup.sh
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

