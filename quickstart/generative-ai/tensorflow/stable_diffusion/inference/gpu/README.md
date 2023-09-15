<!--- 0. Title -->
# Stable Diffusion inference

<!-- 10. Description -->
## Description

This document has instructions for running Stable Diffusion inference using
Intel® Extension for TensorFlow with Intel® Data Center GPU Flex Series.

<!--- 20. GPU Setup -->
## Software Requirements:
- Intel® Data Center GPU Flex Series
- Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
- Follow [instructions](https://pypi.org/project/intel-extension-for-tensorflow) to install the latest ITEX version and other prerequisites.

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

For accuracy measurement, please download the reference data file `img_arrays_for_acc.txt` from the link [here](https://github.com/intel/intel-extension-for-tensorflow/tree/main/examples/stable_diffussion_inference/nv_results). Set the `REFERENCE_RESULT_FILE` to point to the file.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|:-------------:|:-------------:|
| `online_inference` | Runs online inference for FP32 and FP16 precisions on Flex series 170 |
| `accuracy` | Runs batch inference for FP16 precision on Flex series 170 |

<!--- 50. Baremetal -->
## Run the model
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```

### Run the model on Baremetal
Navigate to the model zoo directory, and set environment variables:
```
cd models
export OUTPUT_DIR=<path where output log files will be written>
export PRECISION=< provide fp32 or fp16 as precision input >

# Optional envs
export BATCH_SIZE=<Set batch_size else it will run with default batch of 1>

# Set the following env only for accuracy script
export REFERENCE_RESULT_FILE=<path to reference results file only for accuracy >

Run the model specific dependencies:
NOTE: Installing dependencies in setup.sh may require root privilege
./quickstart/generative-ai/tensorflow/stable_diffusion/inference/gpu/setup.sh

Run quickstart script:
./quickstart/generative-ai/tensorflow/stable_diffusion/inference/gpu/<script_name>.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
