<!--- 0. Title -->
# TensorFlow SSD-MobileNet inference

<!-- 10. Description -->
## Description

This document has instructions for running SSD-MobileNet inference using
Intel-optimized TensorFlow.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run SSD-MobileNet inference in a container.
```
tf-spr-ssd-mobilenet-inference
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-ssd-mobilenet-inference.tar.gz
├──tf-spr-ssd-mobilenet-inference.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32 or bfloat16). Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32 or bfloat16). Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32 or bfloat16). |

<!--- 30. Datasets -->
## Datasets

The [COCO validation dataset](http://cocodataset.org) is used in these
SSD-Mobilenet quickstart scripts. The accuracy quickstart script require the dataset to be converted into the TF records format.
See the [COCO dataset](https://github.com/IntelAI/models/tree/master/datasets/coco) for instructions on
downloading and preprocessing the COCO validation dataset.

Set the `DATASET_DIR` to point to the dataset directory that contains the TF records file `coco_val.record` when running SSD-MobileNet accuracy script.

## Build the container

The SSD-MobileNet inference package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the SSD-MobileNet inference container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-ssd-mobilenet-inference.tar.gz
cd tf-spr-ssd-mobilenet-inference

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-ssd-mobilenet-inference` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run SSD-MobileNet inference in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. 
The dataset is required only for the accuracy script.
By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd tf-spr-ssd-mobilenet-inference

# Set the required environment vars
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# To test accuracy, also specify the dataset directory
export DATASET_DIR=<path to the dataset>
SCRIPT=accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

