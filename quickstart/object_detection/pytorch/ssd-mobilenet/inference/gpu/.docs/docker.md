<!--- 60. Docker -->
## Run the model

Requirements:
* Host machine has Intel® Data Center GPU Flex Series.
* Host machine has the Intel® Data Center GPU Flex Series Ubuntu driver. Please follow the [link](https://registrationcenter.intel.com/en/products/download/4125/) to download.
* Host machine has Docker installed
* Download and build the Intel® Extension for PyTorch (IPEX) container using the [link](https://registrationcenter.intel.com/en/products/subscription/956/).
  (`model-zoo:pytorch-ipex-gpu`)

Prior to building the <model name> <mode> container, ensure that you have
built the IPEX container (`model-zoo:pytorch-ipex-gpu`).

[Extract the package](#model-package), then use the `build.sh`
script to build the container. After the container has been built, you can
run the model <mode> using the `run.sh` script.

The `run.sh` script will execute one of the [quickstart](#quick-start-scripts) script
using the container that was just built. By default, the
`inference_block_format.sh` script will be run. To run a different script,
specify the script name of the quickstart script using the `SCRIPT`
environment variable. See the snippet below for an example.

> Note: Ensure that your system has the proxy environment variables
> set (if needed), otherwise the container build may fail when trying to pull external
> dependencies (like apt-get and pip installs).

The inference scripts will download the model weights from https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth to the anywhere directory you choice, and set environment 'PRETRAINED_MODEL'.
```
# Extract the package
tar -xzf <package name>

# Navigate to the SSD-Mobilenet inference directory
cd pytorch-gpu-ssd-mobilenet-inference

# Set environment vars for the dataset and an output directory
export DATASET_DIR=<Path to the VOC2007 folder>
export OUTPUT_DIR=<Where_to_save_OUTPUT_DIR>
export PRETRAINED_MODEL=<Where_to_load_path>

# Run the container with default script in run.sh (inference_with_dummy_data.sh)
./run.sh
```
