<!--- 0. Title -->
# PyTorch DLRM inference

<!-- 10. Description -->
## Description

This document has instructions for running DLRM inference using
Intel-optimized PyTorch.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run DLRM inference in a container.
```
pytorch-spr-dlrm-inference
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── pytorch-spr-dlrm-inference.tar.gz
├── pytorch-spr-dlrm-inference.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_performance.sh` | Run inference to verify performance for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16). |

> Note: The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets

### Criteo Terabyte Dataset

The [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) is
used to run DLRM. To download the dataset, you will need to visit the Criteo website and accept
their terms of use:
[https://labs.criteo.com/2013/12/download-terabyte-click-logs/](https://labs.criteo.com/2013/12/download-terabyte-click-logs/).
Copy the download URL into the command below as the `<download url>` and
replace the `<dir/to/save/dlrm_data>` to any path where you want to download
and save the dataset.
```
export DATASET_DIR=<dir/to/save/dlrm_data>

mkdir ${DATASET_DIR} && cd ${DATASET_DIR}
curl -O <download url>/day_{$(seq -s , 0 23)}.gz
gunzip day_*.gz
```
The raw data will be automatically preprocessed and saved as `day_*.npz` to
the `DATASET_DIR` when DLRM is run for the first time. On subsequent runs, the
scripts will automatically use the preprocessed data.

## Build the container

The DLRM inference package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the PyTorch/IPEX container as it's base, so ensure that you have built
the `pytorch-ipex-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep pytorch-ipex-spr
model-zoo         pytorch-ipex-spr         f5b473554295        2 hours ago         4.08GB
```

To build the DLRM inference container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf pytorch-spr-dlrm-inference.tar.gz
cd pytorch-spr-dlrm-inference

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:pytorch-spr-dlrm-inference` that will be used to run the model.

## Run the model

Download the DLRM PyTorch weights (`tb00_40M.pt`, 90GB) from the
[MLPerf repo](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch#more-information-about-the-model-weights)
and set the `WEIGHT_PATH` to point to the weights file.
```
export WEIGHT_PATH=<path to the tb00_40M.pt file>
```

After you've downloaded the model weight file and followed the instructions to
[build the container](#build-the-container) and [downloaded dataset](#datasets),
use the `run.sh` script from the container package to run DLRM inference.
Set environment variables to specify the dataset directory, precision to run, and
an output directory. By default, the `run.sh` script will run the
`inference_performance.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd pytorch-spr-dlrm-inference

# Set the required environment vars
export PRECISION=<specify the precision to run>
export WEIGHT_PATH=<path to the tb00_40M.pt file>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_performance.sh quickstart script
./run.sh

# Specify a different quickstart script to run, for example, accuracy.sh
SCRIPT=accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

