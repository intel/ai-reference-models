<!--- 60. Docker -->
## Docker

Requirements:
* Host machine has Intel GPU.
* Host machine has installed Linux kernel that is compatible with GPU drivers.
* Host machine has Docker installed
* Download and build the Intel® Extension for PyTorch (IPEX) container
  (`model-zoo:pytorch-ipex-gpu`)

Prior to building the <model name> <mode> container, ensure that you have
built the IPEX container (`model-zoo:pytorch-ipex-gpu`).

[Extract the package](#model-package), then use the `build.sh`
script to build the container. After the container has been built, you can
run the model <mode> using the `run.sh` script.
Set environment variables for the path to [imagenet dataset](#datasets),
the precision to run, and tan output directory for logs.

The `run.sh` script will execute one of the [quickstart](#quick-start-scripts) script
using the container that was just built. By default, the
`training_block_format.sh` script will be run. To run a different script,
specify the script name of the quickstart script using the `SCRIPT`
environment variable. See the snippet below for an example.

> Note: Ensure that your system has the proxy environment variables
> set (if needed), otherwise the container build may fail when trying to pull external
> dependencies (like apt-get and pip installs).

```
# Extract the package
tar -xzf <package name>
cd <package dir>

# Build the container
./build.sh

# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export PRECISION=<specify the precision to run (fp32 or bf16)>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with the default training_block_format.sh script
./run.sh

# Or, specify a different quickstart script to run with another precision
export PRECISION=<precision to run>
SCRIPT=training_plain_format.sh ./run.sh
```
