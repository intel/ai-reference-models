<!--- 60. Docker -->
## Docker

Requirements:
* Host machine has Intel(R) Data Center GPU Flex Series.
* GPU-compatible drivers need to be installed. Please follow the [link](https://registrationcenter.intel.com/en/products/download/4125/) to download.
* Docker
* Download and build the [Intel® Extension for TensorFlow (ITEX) container](https://registrationcenter.intel.com/en/products/subscription/956/)(`model-zoo:tensorflow-itex-gpu`)

After extracting the <package name>, use the `build.sh`
script to build the container. After the container has been built, you can
run the model <mode> using the `run.sh` script. Set environment variables
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
tar -xzf <package name>
cd <package dir>

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
