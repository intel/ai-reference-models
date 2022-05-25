<!--- 0. Title -->
# PyTorch Mask R-CNN training

<!-- 10. Description -->
## Description

This document has instructions for running Mask R-CNN training using
Intel-optimized PyTorch.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run Mask R-CNN training in a container.
```
pytorch-spr-maskrcnn-training
├── README.md
├── build.sh
├── download_dataset.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── pytorch-spr-maskrcnn-training.tar.gz
├── pytorch-spr-maskrcnn-training.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training.sh` | Runs training for the specified precision (fp32, avx-fp32， bf16, or bf32). |

> Note: The `avx-fp32` precision runs the same scripts as `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets

Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script
from the container package.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset
will be downloaded. This environment variable will be used again when running quickstart scripts.
```
cd pytorch-spr-maskrcnn-training
export DATASET_DIR=<directory where the dataset will be saved>
bash download_dataset.sh
```

## Build the container

The Mask R-CNN training package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the PyTorch/IPEX container as it's base, so ensure that you have built
the `pytorch-ipex-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep pytorch-ipex-spr
model-zoo         pytorch-ipex-spr         f5b473554295        2 hours ago         4.08GB
```

To build the Mask R-CNN training container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf pytorch-spr-maskrcnn-training.tar.gz
cd pytorch-spr-maskrcnn-training

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:pytorch-spr-maskrcnn-training` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container package
to run Mask R-CNN training. Set environment variables to specify the dataset directory,
precision to run, and an output directory. The `run.sh` script will run the
`training.sh` quickstart script in the container.
```
# Navigate to the container package directory
cd pytorch-spr-maskrcnn-training

# Set the required environment vars
export PRECISION=<specify the precision to run>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Optionally set the batch size (default is 112)
export BATCH_SIZE=<batch size>

# Run the container with training.sh quickstart script
./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

