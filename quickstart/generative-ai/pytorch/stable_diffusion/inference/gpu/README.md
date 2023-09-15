<!--- 0. Title -->
# Stable diffusion inference

<!-- 10. Description -->
## Description

This document has instructions for running Stable diffusion inference using
Intel(R) Extension for PyTorch with GPU.

<!--- 20. GPU Setup -->
## Software Requirements:
- Intel® Data Center GPU Flex Series
- Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
- Follow [instructions](https://pypi.org/project/intel-extension-for-pytorch/) to install the latest IPEX version and other prerequisites.

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

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `online_inference.sh` | Inference for specified precision(FP32 or FP16) with batch size 1 on Flex series 170 |

<!--- 50. Baremetal -->
## Run the model
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```
* Navigate to model zoo directory:
  ```bash
  # Navigate to the model zoo repo
  cd models
  ```

### Run the model on Baremetal
Set environment variables to run the quickstart script:
```
export PRECISION=<provide either fp32 or fp16>
export OUTPUT_DIR=<Path to save the output logs>

# Optional envs
export BATCH_SIZE=<Set batch_size else it will run with default batch>

Run the model specific dependencies:
NOTE: Installing dependencies in setup.sh may require root privilege
./quickstart/generative-ai/pytorch/stable_diffusion/inference/gpu/setup.sh

# Run a quickstart script
./quickstart/generative-ai/pytorch/stable_diffusion/inference/gpu/online_inference.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
