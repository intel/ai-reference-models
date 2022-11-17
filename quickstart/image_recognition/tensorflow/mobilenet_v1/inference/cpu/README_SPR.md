<!--- 0. Title -->
# TensorFlow MobileNet V1 inference

<!-- 10. Description -->
## Description

This document has instructions for running MobileNet V1 inference using
Intel-optimized TensorFlow.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run MobileNet V1 inference in a container.
```
tf-spr-mobilenet-v1-inference
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-mobilenet-v1-inference.tar.gz
├──tf-spr-mobilenet-v1-inference.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, int8 or bfloat16) with 1000 steps, 500 warmup steps and `batch-size=1`. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, int8 or bfloat16) with 1000 steps, 500 warmup steps and `batch-size=448`. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bfloat16) with `batch-size=100`. |

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running MobileNet V1.

## Build the container

The MobileNet V1 inference package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the MobileNet V1 inference container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-mobilenet-v1-inference.tar.gz
cd tf-spr-mobilenet-v1-inference

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-mobilenet-v1-inference` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run MobileNet V1 inference in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. 
The dataset is required for accuracy and optional for other inference scripts.
By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd tf-spr-mobilenet-v1-inference

# Set the required environment vars
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>
Set env variable for BATCH_SIZE to run the workload with a different batch size. If not set, the workload will run with the default batch size which gives the best performance.
export BATCH_SIZE=<Mention the batch size>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# To test accuracy, also specify the dataset directory
export DATASET_DIR=<path to the dataset>
export BATCH_SIZE=<Mention the batch size>
SCRIPT=accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

