<!--- 0. Title -->
# ResNet50 v1.5 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 v1.5 inference using
Intel® Extension for TensorFlow with Intel® Data Center GPU Flex Series.

<!--- 20. GPU Setup -->
## Software Requirements:
- Intel® Data Center GPU Flex Series
- Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
- Follow [instructions](https://pypi.org/project/intel-extension-for-tensorflow/) to install the latest ITEX version and other prerequisites.

- Intel® oneAPI Base Toolkit: Need to install components of Intel® oneAPI Base Toolkit
  - Intel® oneAPI DPC++ Compiler
  - Intel® oneAPI Threading Building Blocks (oneTBB)
  - Intel® oneAPI Math Kernel Library (oneMKL)
  - Follow [instructions](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=offline) to download and install the latest oneAPI Base Toolkit.

  - Set environment variables for Intel® oneAPI Base Toolkit: 
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh
    source {ONEAPI_ROOT}/mpi/latest/env/vars.sh
    source {ONEAPI_ROOT}/ccl/latest/env/vars.sh
    ```

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `online_inference` | Runs online inference for int8 precision on Flex series 170 |
| `batch_inference` | Runs batch inference for int8 precision on Flex series 170 |
| `accuracy` | Measures the model accuracy for int8 precision on Flex series 170 |
| `flex_multi_card_online_inference` | Runs online inference for int8 precision on Flex series 140 |
| `flex_multi_card_batch_inference` | Runs batch inference for int8 precision on Flex series 140 |

<!--- 50. Baremetal -->
## Run the model
Install the following pre-requisites:
* Download the frozen graph model file, and set the FROZEN_GRAPH environment variable to point to where it was saved:
  ```bash
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/gpu/resnet50_v1_int8.pb
  ```
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading and preprocessing the ImageNet dataset. The path to the ImageNet
TF records files will need to be set as the `DATASET_DIR` environment variable
prior to running a [quickstart script](#quick-start-scripts).

### Run the model on Baremetal
Navigate to the Model Zoo directory, and set environment variables:
```
cd models
export OUTPUT_DIR=<path where output log files will be written>
export PRECISION=int8
export FROZEN_GRAPH=<path to pretrained model file (*.pb)>
export GPU_TYPE=flex_series

# Optional envs
export BATCH_SIZE=<Set batch_size else it will run with default batch>

# Set 'DATASET_DIR' only when running "accuracy.sh" script:
export DATASET_DIR=<path to the preprocessed imagenet dataset directory>

Run quickstart script:
./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/<script name>.sh
```
To execute `flex_multi_card_batch_inference.sh` and `flex_multi_card_online_inference.sh` on Flex series 140, install the following components 

```bash
apt-get update && \
apt-get install -y --no-install-recommends --fix-missing parallel pciutils numactl 
```
Then execute the quickstart scripts. For batch inference, the default batch size for Flex 140 is 256.
```bash
./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/flex_multi_card_batch_inference.sh 
./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/flex_multi_card_online_inference.sh
```
<!--- 80. License -->
## License

[LICENSE](/LICENSE)
