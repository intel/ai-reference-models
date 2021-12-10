<!--- 0. Title -->
# PyTorch ssd-resnet34 inference

<!-- 10. Description -->
## Description

This document has instructions for running ssd-resnet34 inference using
Intel-optimized PyTorch.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run ssd-resnet34 inference in a container.
```
pytorch-spr-ssd-resnet34-inference
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── pytorch-spr-ssd-resnet34-inference.tar.gz
├── pytorch-spr-ssd-resnet34-inference.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |
| `inference_throughput.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |

> Note: The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets

Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script
from the container package.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset
will be downloaded. This environment variable will be used again when running quickstart scripts.
```
cd pytorch-spr-ssd-resnet34-inference
export DATASET_DIR=<directory where the dataset will be saved>
bash download_dataset.sh
```

## Build the container

The ssd-resnet34 inference package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the PyTorch/IPEX container as it's base, so ensure that you have built
the `pytorch-ipex-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep pytorch-ipex-spr
model-zoo         pytorch-ipex-spr         f5b473554295        2 hours ago         4.08GB
```

To build the ssd-resnet34 inference container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf pytorch-spr-ssd-resnet34-inference.tar.gz
cd pytorch-spr-ssd-resnet34-inference

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:pytorch-spr-ssd-resnet34-inference` that will be used to run the model.

## Run the model

Download the pretrained model weights using the script from the container package
and set the `CHECKPOINT_DIR` environment variable to point to the downloaded file:
```
cd pytorch-spr-ssd-resnet34-inference
export CHECKPOINT_DIR=<directory where to save the pretrained model>
sh download_model.sh
```

After downloading the pretrained model and following the instructions to
[build the container](#build-the-container) and [prepare the dataset](#datasets),
use the `run.sh` script from the container package to run ssd-resnet34 inference
using docker. Set environment variables to specify the dataset directory,
precision to run, and an output directory for logs. By default, the `run.sh`
script will run the `inference_realtime.sh` quickstart script. To run a different
script, specify the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd pytorch-spr-ssd-resnet34-inference

# Set the required environment vars
export DATASET_DIR=<path to the coco dataset>
export CHECKPOINT_DIR=<path to the downloaded weights directory>
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# To run a difference quickstart script, us the SCRIPT env var
SCRIPT=accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

