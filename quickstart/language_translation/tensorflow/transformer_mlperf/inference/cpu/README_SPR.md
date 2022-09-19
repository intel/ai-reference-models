<!--- 0. Title -->
# TensorFlow Transformer Language inference

<!-- 10. Description -->
## Description

This document has instructions for running Transformer Language inference using
Intel-optimized TensorFlow.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run Transformer Language inference in a container.
```
tf-spr-transformer-mlperf-inference
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-transformer-mlperf-inference.tar.gz
├──tf-spr-transformer-mlperf-inference.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference (batch-size=1) using 4 cores per instance for the specified precision (fp32 or bfloat16). |
| `inference_throughput.sh` | Runs multi instance batch inference (batch-size=64 for the precisions fp32 or bfloat16, and batch-size=448 for int8 precision) using 1 instance per socket. |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32 or bfloat16). |

<!--- 30. Datasets -->
## Datasets

Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/transformer_data/README.md) to download and preprocess the WMT English-German dataset.
Set `DATASET_DIR` to point out to the location of the dataset directory.

## Build the container

The Transformer Language inference package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the Transformer Language inference container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-transformer-mlperf-inference.tar.gz
cd tf-spr-transformer-mlperf-inference

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-transformer-mlperf-inference` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run Transformer Language inference in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. 
By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd tf-spr-transformer-mlperf-inference

# Set the required environment vars
export PRECISION=<specify the precision to run>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# To run a different script, specify the name of the script using the `SCRIPT` environment variable
SCRIPT=accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

