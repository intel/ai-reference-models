<!--- 0. Title -->
# TensorFlow SSD-ResNet34 inference

<!-- 10. Description -->
## Description

This document has instructions for running SSD-ResNet34 inference using
Intel-optimized TensorFlow.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run SSD-ResNet34 inference in a container.
```
tf-spr-ssd-resnet34-inference
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-ssd-resnet34-inference.tar.gz
├──tf-spr-ssd-resnet34-inference.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference (batch-size=1) using 4 cores per instance for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput.sh` | Runs multi instance batch inference (batch-size=16) using 1 instance per socket for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bfloat16). |

<!--- 30. Datasets -->
## Datasets

The SSD-ResNet34 accuracy script `accuracy.sh` uses the
[COCO validation dataset](http://cocodataset.org) in the TF records
format. See the [COCO dataset document](https://github.com/IntelAI/models/tree/master/datasets/coco) for
instructions on downloading and preprocessing the COCO validation dataset.
The inference scripts use synthetic data, so no dataset is required.

After the script to convert the raw images to the TF records file completes, rename the tf_records file:
```
mv ${OUTPUT_DIR}/coco_val.record ${OUTPUT_DIR}/validation-00000-of-00001
```
Set the `DATASET_DIR` to the folder that has the `validation-00000-of-00001`
file when running the accuracy test. Note that the inference performance
test uses synthetic dataset.

## Build the container

The SSD-ResNet34 inference package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the SSD-ResNet34 inference container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-ssd-resnet34-inference.tar.gz
cd tf-spr-ssd-resnet34-inference

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-ssd-resnet34-inference` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run SSD-ResNet34 inference in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. 
The dataset is required for the accuracy script.
By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd tf-spr-ssd-resnet34-inference

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

