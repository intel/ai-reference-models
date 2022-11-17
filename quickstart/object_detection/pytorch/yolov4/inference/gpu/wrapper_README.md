<!--- 0. Title -->
# YOLOv4 inference

<!-- 10. Description -->
## Description

This document has instructions for running YOLOv4 inference using
Intel(R) Extension for PyTorch with GPU.

<!--- 20. Model package -->
## Model Package

The model package includes the scripts and libraries needed to
build and run YOLOv4 inference using a docker container. Note that
this model container uses the PyTorch IPEX GPU container as it's base,
and it requires the `model-zoo:pytorch-ipex-gpu` image to be built before
the model container is built.
```
pytorch-gpu-yolov4-inference
├── build.sh
├── info.txt
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── pytorch-gpu-yolov4-inference.tar.gz
├── pytorch-gpu-yolov4-inference.Dockerfile
├── README.md
└── run.sh
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
| `inference_with_dummy_data.sh` | Inference with int8 batch_size64 dummy data |

<!--- 60. Docker -->
## Docker

Requirements:
* Host machine has Intel® Data Center GPU Flex Series.
* Host machine has the Intel® Data Center GPU Flex Series Ubuntu driver. Please follow the [link](https://registrationcenter.intel.com/en/products/download/4125/) to download.
* Host machine has Docker installed
* Download and build the Intel® Extension for PyTorch (IPEX) container using the [link](https://registrationcenter.intel.com/en/products/subscription/956/).
  (`model-zoo:pytorch-ipex-gpu`)

Prior to building the YOLOv4 inference container, ensure that you have
built the IPEX container (`model-zoo:pytorch-ipex-gpu`).

[Extract the package](#model-package), then use the `build.sh`
script to build the container. After the container has been built, you can
run the model inference using the `run.sh` script.

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
tar -xzf pytorch-gpu-yolov4-inference.tar.gz

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

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

