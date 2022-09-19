<!--- 0. Title -->
# TensorFlow SSD-ResNet34 training

<!-- 10. Description -->

This document has instructions for running SSD-ResNet34 training
using Intel-optimized TensorFlow.


## Model Package

The model package includes the Dockerfile and scripts needed to build and
run SSD-ResNet34 training in a container.
```
tf-spr-ssd-resnet34-training
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-ssd-resnet34-training.tar.gz
├──tf-spr-ssd-resnet34-training.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training.sh` | Uses mpirun to execute 2 processes with 1 process per socket with a batch size of 56 for the specified precision (fp32 or bfloat16). Logs for each instance are saved to the output directory. |

<!--- 30. Datasets -->
## Datasets

SSD-ResNet34 training uses the COCO training dataset. Use the [instructions](https://github.com/IntelAI/models/tree/master/datasets/coco/README_train.md) to download and preprocess the dataset.


## Build the container

The SSD-ResNet34 training package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the SSD-ResNet34 training container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-ssd-resnet34-training.tar.gz
cd tf-spr-ssd-resnet34-training

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-ssd-resnet34-training` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run SSD-ResNet34 training in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. By default, the `run.sh` script will run the
`training.sh` quickstart script.
```
# Navigate to the container package directory
cd tf-spr-ssd-resnet34-training

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

