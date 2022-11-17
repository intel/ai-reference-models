<!--- 0. Title -->
# SSD-MobileNet inference

<!-- 10. Description -->
## Description

This document has instructions for running SSD-MobileNet inference using
Intel(R) Extension for TensorFlow with Intel(R) Data Center GPU Flex Series.

<!--- 20. Model package -->
## Model Package

The model package includes the scripts and libraries needed to
build and run SSD-MobileNet inference using a docker container. Note that
this model container uses the Tensorflow ITEX GPU container as it's base,
and it requires the `model-zoo:tensorflow-itex-gpu` image to be built before
the model container is built.
```
tf-atsm-ssd-mobilenet-inference
├── build.sh
├── info.txt
├── licenses
│   ├── LICENSE
│   └── third_party
│       ├── Intel_Model_Zoo_v2.0_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.0_ML_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.3_PyTorch.txt
│       └── licenses.txt
├── model_packages
│   └── tf-atsm-ssd-mobilenet-inference.tar.gz
├── README.md
├── run.sh
└── tf-atsm-ssd-mobilenet-inference.Dockerfile
```


<!--- 30. Datasets -->
## Datasets

Download and preprocess the COCO dataset using the [instructions here](/datasets/coco/README.md).
After running the conversion script you should have a directory with the
COCO dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running SSD-MobileNet.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| [`online_inference.sh`](online_inference.sh) | Runs online inference for int8 precision |
| [`batch_inference.sh`](batch_inference.sh) | Runs batch inference for int8 precision |
| [`accuracy.sh`](accuracy.sh) | Measures the model accuracy for int8 precision  |

<!--- 60. Docker -->
## Docker

Requirements:
* Host machine has Intel(R) Data Center GPU Flex Series.
* GPU-compatible drivers need to be installed. Please follow the [link](https://registrationcenter.intel.com/en/products/download/4125/) to download.
* Docker
* Download and build the [Intel® Extension for TensorFlow (ITEX) container](https://registrationcenter.intel.com/en/products/subscription/956/)(`model-zoo:tensorflow-itex-gpu`)

After extracting the tf-atsm-ssd-mobilenet-inference.tar.gz, use the `build.sh`
script to build the container. After the container has been built, you can
run the model inference using the `run.sh` script. Set environment variables
for DATASET_DIR to the path to COCO TF records files,if you do not set it 
dummy data will be used. However, for accuracy you have to provide the real dataset.
Set OUTPUT_DIR to an output directory where log files will be written. 

The `run.sh` script will execute one of the [quickstart](#quickstart) script
using the container that was just built. By default, the
`quickstart/online_inference.sh` script will be run. To change which
script gets run, either edit the `run.sh` file, or specify the name of file
to run using the `SCRIPT` environment variable.
See the snippet below for an example on how to do this.

> Note: Ensure that your system has the proxy environment variables
> set, otherwise the container build may fail when trying to pull external
> dependencies (like apt-get and pip installs).

```
#Extract the package
tar -xzf tf-atsm-ssd-mobilenet-inference.tar.gz
cd tf-atsm-ssd-mobilenet-inference

# Build the container
./build.sh

#Set environment variables
export PRECISION=int8
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with the default online_inference.sh script
./run.sh

# Specify a different quickstart script to run
SCRIPT=quickstart/batch_inference.sh ./run.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

