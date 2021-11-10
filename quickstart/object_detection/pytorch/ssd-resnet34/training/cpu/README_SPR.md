<!--- 0. Title -->
# PyTorch ssd-resnet34 training

<!-- 10. Description -->
## Description

This document has instructions for running ssd-resnet34 training using
Intel-optimized PyTorch.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run ssd-resnet34 training in a container.
```
pytorch-spr-ssd-resnet34-training
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── pytorch-spr-ssd-resnet34-training.tar.gz
├── pytorch-spr-ssd-resnet34-training.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `train_performance.sh` | Tests the training performance for SSD-ResNet34 for the specified precision (fp32, avx-fp32, or bf16). |
| `train_accuracy.sh` | Tests the training accuracy for SSD-ResNet34 for the specified precision (fp32, avx-fp32, or bf16). |

> Note: The `avx-fp32` precision runs the same scripts as `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.


## Datasets

### COCO

The [COCO dataset](https://cocodataset.org) is used to run ssd-resnet34.

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
is the directory that should be used when setting the `DATASET_DIR` environment
variable for ssd-resnet34 (for example: `export DATASET_DIR=/home/<user>/coco`).

## Build the container

The ssd-resnet34 training package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the PyTorch/IPEX container as it's base, so ensure that you have built
the `pytorch-ipex-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep pytorch-ipex-spr
model-zoo         pytorch-ipex-spr         f5b473554295        2 hours ago         4.08GB
```

To build the ssd-resnet34 training container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf pytorch-spr-ssd-resnet34-training.tar.gz
cd pytorch-spr-ssd-resnet34-training

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:pytorch-spr-ssd-resnet34-training` that will be used to run the model.

## Run the model

Download the backbone weights and set the `BACKBONE_WEIGHTS` environment variable
to point to the downloaded file:
```
curl -O https://download.pytorch.org/models/resnet34-333f7ec4.pth
export BACKBONE_WEIGHTS=$(pwd)/resnet34-333f7ec4.pth
```

After you've downloaded the backbone weights and followed the instructions to
[build the container](#build-the-container) and [prepare the dataset](#datasets),
use the `run.sh` script from the container package to run ssd-resnet34 training
using docker. Set environment variables to point to the COCO dataset directory,
weights, precision, and an output directory for logs. By default, the `run.sh`
script will run the `train_performance.sh` quickstart script. To run the `train_accuracy.sh`
script instead, specify that script name using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd pytorch-spr-ssd-resnet34-training

# Set the required environment vars
export DATASET_DIR=<path to the COCO dataset directory>
export BACKBONE_WEIGHTS=<path to the downloaded weights file>
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with train_performance.sh quickstart script
./run.sh

# Run a different script by specifying the SCRIPT env var
SCRIPT=train_accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

