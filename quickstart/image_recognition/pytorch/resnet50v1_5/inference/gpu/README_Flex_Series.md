<!--- 0. Title -->
# ResNet50v1.5 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50v1.5 inference using
Intel(R) Extension for PyTorch with GPU.

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
| `inference_block_format.sh` | Runs ResNet50 inference (block format) for the specified precision (int8) on Flex series 170 |
| `flex_multi_card_batch_inference.sh` | Runs ResNet50 inference (block format) for the specified precision (int8) and batch size on Flex series 140|
| `flex_multi_card_online_inference.sh` | Runs Online ResNet50 inference (block format) for the specified precision (int8) on Flex series 140| 

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
* Navigate to ResNet50v1.5 inference directory:
  ```bash
  # Navigate to the model zoo repo
  cd models
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading and preprocessing the ImageNet dataset. The path to the ImageNet
dataset files will need to be set as the `DATASET_DIR` environment variable
prior to running a [quickstart script](#quick-start-scripts).

### Run the model on Baremetal
Set environment variables for the path to your dataset, an output directory to run the quickstart script:
```
To run with ImageNet data, the dataset directory will need to be specified in addition to an output directory and precision.
export DATASET_DIR=<path to the preprocessed imagenet dataset>
export OUTPUT_DIR=<Path to save the output logs>

# Optional envs
export BATCH_SIZE=<Set batch_size else it will run with default batch>

# Run a quickstart script
./quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/inference_block_format.sh
```
To execute `flex_multi_card_batch_inference.sh` and `flex_multi_card_online_inference.sh` on Flex series 140, install the following components 

```bash
apt-get update && \
apt-get install -y --no-install-recommends --fix-missing parallel pciutils numactl 
```
Then execute the quickstart scripts. For batch inference, the default batch size for Flex 140 is 256.
```bash
./quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/flex_multi_card_batch_inference.sh 
./quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/flex_multi_card_online_inference.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
