<!--- 0. Title -->
# SSD-Mobilenetv1 inference

<!-- 10. Description -->
## Description

This document has instructions for running SSD-Mobilenetv1 inference using
Intel(R) Extension for PyTorch with GPU.

<!--- 20. Model package -->
## Model Package

The model package includes the scripts and libraries needed to
build and run SSD-Mobilenetv1 inference using a docker container. Note that
this model container uses the PyTorch IPEX GPU container as it's base,
and it requires the `model-zoo:pytorch-ipex-gpu` image to be built before
the model container is built.
```
pytorch-gpu-ssd-mobilenet-inference
├── build.sh
├── info.txt
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── pytorch-gpu-ssd-mobilenet-inference.tar.gz
├── pytorch-gpu-ssd-mobilenet-inference.Dockerfile
├── README.md
└── run.sh
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

<!--- 60. Docker -->
## Run the model

Requirements:
* Host machine has Intel® Data Center GPU Flex Series.
* Host machine has the Intel® Data Center GPU Flex Series Ubuntu driver. Please follow the [link](https://registrationcenter.intel.com/en/products/download/4125/) to download.
* Host machine has Docker installed
* Download and build the Intel® Extension for PyTorch (IPEX) container using the [link](https://registrationcenter.intel.com/en/products/subscription/956/).
  (`model-zoo:pytorch-ipex-gpu`)

Prior to building the SSD-Mobilenetv1 inference container, ensure that you have
built the IPEX container (`model-zoo:pytorch-ipex-gpu`).

[Extract the package](#model-package), then use the `build.sh`
script to build the container. After the container has been built, you can
run the model inference using the `run.sh` script.

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
tar -xzf pytorch-gpu-ssd-mobilenet-inference.tar.gz

# Navigate to the SSD-Mobilenet inference directory
cd pytorch-gpu-ssd-mobilenet-inference

# Set environment vars for the dataset and an output directory
export DATASET_DIR=<Path to the VOC2007 folder>
export OUTPUT_DIR=<Where_to_save_OUTPUT_DIR>
export PRETRAINED_MODEL=<Where_to_load_path>

# Run the container with default script in run.sh (inference_with_dummy_data.sh)
./run.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

