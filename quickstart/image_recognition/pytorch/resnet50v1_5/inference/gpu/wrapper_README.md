<!--- 0. Title -->
# ResNet50v1.5 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50v1.5 inference using
Intel(R) Extension for PyTorch with GPU.

<!--- 20. Model package -->
## Model Package

The model package includes the scripts and libraries needed to
build and run ResNet50v1.5 inference using a docker container. Note that
this model container uses the PyTorch IPEX GPU container as it's base,
and it requires the `model-zoo:pytorch-ipex-gpu` image to be built before
the model container is built.
```
pytorch-gpu-resnet50v1-5-inference
├── build.sh
├── info.txt
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── pytorch-gpu-resnet50v1-5-inference.tar.gz
├── pytorch-gpu-resnet50v1-5-inference.Dockerfile
├── README.md
└── run.sh
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
| inference_block_format.sh | Runs ResNet50 inference (block format) for the specified precision (int8) |

<!--- 60. Docker -->
## Docker

Requirements:
* Host machine has Intel® Data Center GPU Flex Series.
* Host machine has the Intel® Data Center GPU Flex Series Ubuntu driver. Please follow the [link](https://registrationcenter.intel.com/en/products/download/4125/) to download.
* Host machine has Docker installed.
* Download and build the Intel® Extension for PyTorch (IPEX) container using the [link](https://registrationcenter.intel.com/en/products/subscription/956/).
  (`model-zoo:pytorch-ipex-gpu`)

Prior to building the ResNet50v1.5 inference container, ensure that you have
built the IPEX container (`model-zoo:pytorch-ipex-gpu`).

[Extract the package](#model-package), then use the `build.sh`
script to build the container. After the container has been built, you can
run the model inference using the `run.sh` script.
Set environment variables for the path to [imagenet dataset](#datasets),
the precision to run, and tan output directory for logs.

The `run.sh` script will execute one of the [quickstart](#quick-start-scripts) script
using the container that was just built. By default, the
`inference_block_format.sh` script will be run. To run a different script,
specify the script name of the quickstart script using the `SCRIPT`
environment variable. See the snippet below for an example.

> Note: Ensure that your system has the proxy environment variables
> set (if needed), otherwise the container build may fail when trying to pull external
> dependencies (like apt-get and pip installs).

```
# Extract the package
tar -xzf pytorch-gpu-resnet50v1-5-inference.tar.gz
cd pytorch-gpu-resnet50v1-5-inference

# Build the container
./build.sh

# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export PRECISION=int8
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with the default inference_block_format.sh script
./run.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

