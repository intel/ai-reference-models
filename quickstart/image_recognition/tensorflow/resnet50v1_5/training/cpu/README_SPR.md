<!--- 0. Title -->
# TensorFlow ResNet50 v1.5 training

<!-- 10. Description -->

This document has instructions for running ResNet50 v1.5 training
using Intel-optimized TensorFlow.


## Model Package

The model package includes the Dockerfile and scripts needed to build and
run ResNet50 v1.5 training in a container.
```
tf-spr-resnet50v1-5-training
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-resnet50v1-5-training.tar.gz
├──tf-spr-resnet50v1-5-training.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `multi_instance_training.sh` | Uses mpirun to execute 1 process per socket with a batch size of 1024 for the specified precision (fp32 or bfloat16 or bfloat32). Checkpoint files and logs for each instance are saved to the output directory. |

<!--- 30. Datasets -->
## Datasets

Note that the ImageNet dataset is used in these ResNet50 v1.5 examples.
Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to this directory when running ResNet50 v1.5.

## Build the container

The ResNet50 v1.5 training package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the ResNet50 v1.5 training container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-resnet50v1-5-training.tar.gz
cd tf-spr-resnet50v1-5-training

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-resnet50v1-5-training` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run ResNet50 v1.5 training in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. By default, the `run.sh` script will run the
`multi_instance_training.sh` quickstart script.
```
# Navigate to the container package directory
cd tf-spr-resnet50v1-5-training

# Set the required environment vars
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with multi_instance_training.sh quickstart script
./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

