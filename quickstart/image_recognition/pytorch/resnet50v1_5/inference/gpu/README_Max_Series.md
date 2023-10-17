<!--- 0. Title -->
# ResNet50v1.5 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50v1.5 inference using
Intel® Extension for PyTorch with GPU.

<!--- 20. GPU Setup -->
## Software Requirements:
- Host machine has Intel® Data Center Max Series 1550 x4 OAM GPU
- Follow instructions to install GPU-compatible driver [647](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html)
- Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
- Follow [instructions](https://pypi.org/project/intel-extension-for-pytorch/) to install the latest IPEX version and other prerequisites.

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
* Clone the AI Reference Models repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```
* Navigate to models directory:
  ```bash
  cd models
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
export PRECISION=int8
export NUM_OAM=<provide 4 for number of OAM Modules supported by the platform>

# Optional envs
export BATCH_SIZE=<Set batch_size else it will run with default batch of 1024>
export NUM_ITERATIONS=<export number of iterations,default is 500>

# Run a quickstart script
./quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/inference_block_format.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
