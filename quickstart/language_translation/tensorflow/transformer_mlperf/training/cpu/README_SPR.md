<!--- 0. Title -->
# TensorFlow Transformer Language training

<!-- 10. Description -->

This document has instructions for running Transformer Language training
using Intel-optimized TensorFlow.


## Model Package

The model package includes the Dockerfile and scripts needed to build and
run Transformer Language training in a container.
```
tf-spr-transformer-mlperf-training
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-transformer-mlperf-training.tar.gz
├──tf-spr-transformer-mlperf-training.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training.sh` | Uses mpirun to execute 2 processes with 1 process per socket with a batch size of 5120 for the specified precision (fp32 or bfloat16). Logs for each instance are saved to the output directory. |

<!--- 30. Datasets -->
## Datasets

Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/transformer_data/README.md) to download and preprocess the WMT English-German dataset.
Set `DATASET_DIR` to point out to the location of the dataset directory.

## Build the container

The Transformer Language training package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the Transformer Language training container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-transformer-mlperf-training.tar.gz
cd tf-spr-transformer-mlperf-training

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-transformer-mlperf-training` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run Transformer Language training in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. By default, the `run.sh` script will run the
`training.sh` quickstart script.
```
# Navigate to the container package directory
cd tf-spr-transformer-mlperf-training

# Set the required environment vars
export PRECISION=<specify the precision to run>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with training.sh quickstart script
./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

