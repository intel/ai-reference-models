<!--- 0. Title -->
# ResNet50v1.5 training

<!-- 10. Description -->
## Description

This document has instructions to run ResNet50v1.5 training using Intel® Extension for PyTorch for GPU.

<!--- 20. GPU Setup -->
## Software Requirements:
- Intel® Data Center GPU Max Series
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
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```
* Navigate to ResNet50v1.5 inference directory and install model specific dependencies for the workload:
  ```bash
  # Navigate to the model zoo repo
  cd models
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
