<!--- 0. Title -->
# TensorFlow MLPerf 3D U-Net inference

<!-- 10. Description -->
## Description

This document has instructions for running MLPerf 3D U-Net inference using
Intel-optimized TensorFlow.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run MLPerf 3D U-Net inference in a container.
```
tf-spr-3d-unet-mlperf-inference
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-3d-unet-mlperf-inference.tar.gz
├──tf-spr-3d-unet-mlperf-inference.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs realtime inference using a default `batch_size=1` for the specified precision (int8, fp32 or bfloat16). To run inference for throughtput, set `BATCH_SIZE` environment variable. |
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (int8, fp32 or bfloat16) with 100 steps and 50 warmup steps. Dummy data is used for performance evaluation. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (int8, fp32 or bfloat16) with 100 steps and 50 warmup steps. Dummy data is used for performance evaluation. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (int8, fp32 or bfloat16). |

<!--- 30. Datasets -->
## Datasets

Download [Brats 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) separately and unzip the dataset.

Set the `DATASET_DIR` to point to the directory that contains the dataset files when running MLPerf 3D U-Net accuracy script.

## Build the container

The MLPerf 3D U-Net inference package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr-3dunet-base.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr-3dunet-base
model-zoo     tensorflow-spr-3dunet-base    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the MLPerf 3D U-Net inference container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-3d-unet-mlperf-inference.tar.gz
cd tf-spr-3d-unet-mlperf-inference

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-3d-unet-mlperf-inference` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run MLPerf 3D U-Net inference in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. 
The dataset is required for running accuracy.
By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd tf-spr-3d-unet-mlperf-inference

# Set the required environment vars
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# To test accuracy, also specify the dataset directory
export DATASET_DIR=<path to the dataset>
SCRIPT=accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

