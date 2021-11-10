<!--- 0. Title -->
# PyTorch RNN-T inference

<!-- 10. Description -->
## Description

This document has instructions for running RNN-T inference using
Intel-optimized PyTorch.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run RNN-T inference in a container.
```
pytorch-spr-rnnt-inference
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── pytorch-spr-rnnt-inference.tar.gz
├── pytorch-spr-rnnt-inference.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `download_inference_dataset.sh` | Download and prepare the LibriSpeech inference dataset. See the [datasets section](#datasets) for instructions on it's usage. |
| `inference_realtime.sh` | Runs multi-instance inference using 4 cores per instance for the specified precision (fp32, avx-fp32, or bf16). |
| `inference_throughput.sh` | Runs multi-instance inference using 1 instance per socket for the specified precision (fp32, avx-fp32, or bf16). |
| `accuracy.sh` | Runs an inference accuracy test for the specified precision (fp32, avx-fp32, or bf16). |

> Note: The `avx-fp32` precision runs the same scripts as `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Build the container

The RNN-T inference package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the PyTorch/IPEX container as it's base, so ensure that you have built
the `pytorch-ipex-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep pytorch-ipex-spr
model-zoo         pytorch-ipex-spr         f5b473554295        2 hours ago         4.08GB
```

To build the RNN-T inference container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf pytorch-spr-rnnt-inference.tar.gz
cd pytorch-spr-rnnt-inference

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:pytorch-spr-rnnt-inference` that will be used to run the model.

## Datasets

The LibriSpeech dataset is used by RNN-T. Use the RNN-T inference container
to download and prepare the inference dataset. Specify a directory for the dataset to be
downloaded to when running the container:
```
export DATASET_DIR=<folder where the inference dataset will be downloaded>
mkdir -p $DATASET_DIR

docker run --rm \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  -w /workspace/pytorch-spr-rnnt-inference \
  -it \
  model-zoo:pytorch-spr-rnnt-inference \
  /bin/bash quickstart/download_inference_dataset.sh
```

This `DATASET_DIR` environment variable will be used again when
[running the model](#run-the-model).

## Run the model

Download the pretrained model and set the `PRETRAINED_MODEL` environment variable
to point to the file:
```
wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O rnnt.pt
export PRETRAINED_MODEL=$(pwd)/rnnt.pt
```

After you've downloaded the pretrained model and followed the instructions to
[build the container](#build-the-container) and [prepare the dataset](#datasets),
use the `run.sh` script from the container package to run RNN-T inference in
docker. Set environment variables to specify the dataset directory, precision to run,
and an output directory. By default, the `inference_realtime.sh` quickstart script will
be run. To run a different quickstart script, set the `SCRIPT` environment variable
to the script of your choice.

The snippet below demonstrates how to run RNN-T inference:
```
# Navigate to the container package directory
cd pytorch-spr-rnnt-inference

# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export PRETRAINED_MODEL=<path to the rnnt.pt file>
export PRECISION=<specify the precision to run (fp32 or bf16)>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# Run a different quickstart script
SCRIPT=accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

