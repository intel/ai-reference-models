<!--- 0. Title -->
# PyTorch RNN-T training

<!-- 10. Description -->
## Description

This document has instructions for running RNN-T training using
Intel-optimized PyTorch.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run RNN-T training in a container.
```
pytorch-spr-rnnt-training
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── pytorch-spr-rnnt-training.tar.gz
├──pytorch-spr-rnnt-training.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `download_dataset.sh` | Download and prepare the LibriSpeech training dataset |
| `training.sh` | Runs RNN-T training for the specified precision (fp32, avx-fp32, or bf16). |

> Note: The `avx-fp32` precision runs the same scripts as `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Build the container

The RNN-T training package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the PyTorch/IPEX container as it's base, so ensure that you have built
the `pytorch-ipex-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep pytorch-ipex-spr
model-zoo         pytorch-ipex-spr         f5b473554295        2 hours ago         4.08GB
```

To build the RNN-T training container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf pytorch-spr-rnnt-training.tar.gz
cd pytorch-spr-rnnt-training

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:pytorch-spr-rnnt-training` that will be used to run the model.

## Datasets

The LibriSpeech dataset is used by RNN-T. Use the RNN-T training container
to download and prepare the training dataset. Specify a directory the dataset will be
downloaded to when running the container:

```
export DATASET_DIR=<folder where the training dataset will be downloaded>
mkdir -p $DATASET_DIR

docker run --rm \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  -w /workspace/pytorch-spr-rnnt-training \
  -it \
  model-zoo:pytorch-spr-rnnt-training \
  /bin/bash quickstart/download_dataset.sh
```

This `DATASET_DIR` environment variable will be used again when
[running the model](#run-the-model).

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container package
to run RNN-T training using docker. Set environment variables to specify the dataset
directory, precision to run, and an output directory. The `run.sh` script will execute the
`training.sh` script in the container that was built.
```
# Navigate to the container package directory
cd pytorch-spr-rnnt-training

# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export PRECISION=<specify the precision to run (fp32 or bf16)>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container using the training.sh script
./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

