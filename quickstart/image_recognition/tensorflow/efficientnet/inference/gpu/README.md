<!--- 0. Title -->
# EfficientNet inference

<!-- 10. Description -->
## Description

This document has instructions for running EfficientNet inference using
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

The [ImageNet](http://www.image-net.org/) validation dataset is used.

Download and extract the ImageNet2012 dataset from http://www.image-net.org/, then move validation images to labeled subfolders, using [the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

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
The folder that contains the `val` directory should be set as the`IMAGE_FILE`
(for example: `export IMAGE_FILE=/home/<user>/imagenet/ILSVRC2012_val_00006697.JPEG`).

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `batch_inference` | Runs EfficientNet B0,B3 batch inference for fp16 precision on Flex series 170 |

<!--- 50. Baremetal -->
## Run the model
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading and preprocessing the ImageNet dataset. The path to the ImageNet
will need to be set as the `DATASET_DIR` environment variable
prior to running a [quickstart script](#quick-start-scripts).

### Run the model on Baremetal
Navigate to the Model Zoo directory, and set environment variables:
```
cd models
export OUTPUT_DIR=<path where output log files will be written>
export MODEL_NAME=<EfficientNetB0 or EfficientNetB3>
export PRECISION=fp16
export IMAGE_FILE=<path to ImageNet Image file>

# Optional envs
export BATCH_SIZE=<Set batch_size else it will run with default batch of 64>

Run quickstart script:
./quickstart/image_recognition/tensorflow/efficientnet/inference/gpu/batch_inference.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
