<!--- 0. Title -->
# PyTorch DLRM Training for Intel(R) Data Center GPU Max Series

<!-- 10. Description -->
## Description

This document has instructions for running DLRM training with BFloat16 precision on Intel® Data Center GPU Max Series using Intel® Extension for PyTorch.

<!--- 20. GPU Setup -->
## Software Requirements:
- Host machine has Intel® Data Center Max Series 1550 x4 OAM GPU
- Follow instructions to install GPU-compatible driver [647](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html)
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

<!--- 30. Datasets -->
## Datasets

The dataset required to train the model is the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf). Please refer to the [link](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm#create-the-synthetic-multi-hot-dataset) to follow steps 1-3. After dataset pre-processing, set the `DATASET_DIR` environment variable to point to the dataset directory. Please note that the pre-processing step requires 700GB of RAM and takes 1-2 days to run.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
|`multi_card_distributed_train.sh` | Runs DLRM distributed training on single-node x4 OAM Modules |

<!--- 50. Baremetal -->
## Run the model
* Clone the AI Reference Models repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```
* Navigate models directory and install model specific dependencies for the workload:
  ```bash
  cd models
  # Install model specific dependencies:
  ./quickstart/recommendation/pytorch/torchrec_dlrm/training/gpu/setup.sh
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading the dataset.

```
# Set environment vars:
export DATASET_DIR=<provide path to the Terabyte Dataset directory>
export NUM_OAM=<provide 4 for number of OAM Modules supported by the platform>
export OUTPUT_DIR=<path to output directory to view logs>
export PRECISION=<provide suitable precision, currently supports bf16 precision> 

# Optional envs
export GLOBAL_BATCH_SIZE=<provide suitable batch size. Default is 65536>
export TOTAL_TRAINING_SAMPLES=<provide suitable number. Default is 4195197692>

# Run a quickstart script
./quickstart/recommendation/pytorch/torchrec_dlrm/training/gpu/multi_card_distributed_train.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
