<!--- 60. Docker -->
## Docker

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
`inference_with_dummy_data.sh` script will be run. To run a different script,
specify the script name of the quickstart script using the `SCRIPT`
environment variable. See the snippet below for an example.

> Note: Ensure that your system has the proxy environment variables
> set (if needed), otherwise the container build may fail when trying to pull external
> dependencies (like apt-get and pip installs). 

You need download pretrained weights from:
yolov4.pth(https://pan.baidu.com/s/1ZroDvoGScDgtE1ja_QqJVw Extraction code:xrq9) or
yolov4.pth(https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
to the anywhere directory you choice, and set environment 'PRETRAINED_MODEL'.
```
# Extract the package
tar -xzf <package name>

# Navigate to the YOLOv4 inference directory
cd pytorch-gpu-YOLOv4-inference

# Build the container
./build.sh

# Set environment vars for pretrained model
export PRETRAINED_MODEL=<Path of the downloaded pretrained-model file on the machine >

# Run the container with the default inference_with_dummy_data.sh script
./run.sh

For inference_with_dummy_data.sh environment is enough, but for inference.sh also need:
export image=<path to the image .jpg file>
export size=<Image size>
export PRECISION=int8
```
