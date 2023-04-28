<!--- 0. Title -->
# BERT Large training for Intel(R) Data Center GPU Max Series

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large training using
Intel-optimized PyTorch with Intel(R) Data Center GPU Max Series.

<!--- 20. GPU Setup -->
## Software Requirements:
- Intel® Data Center GPU Max Series
- Follow [instructions](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html) to install the latest IPEX version and other prerequisites.

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
### Download and Extract the Dataset
Download the [MLCommons BERT Dataset](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v) and download the `results_text.tar.gz` file. Extract the file. After this step, you should have a directory called `results4` that contains 502 files with a total size of 13GB. Set the `DATASET_DIR` to point the location of this dataset. The script assumes the `DATASET_DIR` to be the current working directory. 

The above step is optional. If the `results4` folder is not present in the `DATASET_DIR` path, the quick start scripts automatically download it. 

### Generate the BERT Input Dataset 
The Training script processes the raw dataset. The processed dataset occupies about `539GB` worth of disk space. Additionally, this step can take several hours to complete to generate a folder `hdf5_seq_512`. Hence, the script provides the ability to process the data only once and this data can be volume mounted to the container for future use. Set the `PROCESSED_DATASET_DIR` to point to the location of `hdf5_seq_512`. 

The script assumes the `PROCESSED_DATASET_DIR` to be the current working directory. If the processed folder `hdf5_seq_512` does not exist in the `PROCESSED_DATASET_DIR` path, the quick start scripts process the data.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `bf16_training_plain_format.sh` | Runs BERT Large BF16 training (plain format) on two tiles |
| `ddp_bf16_training_plain_format.sh` | Runs BERT Large Distributed Data Parallel BF16 Training on two tiles | 

<!--- 50. Baremetal -->
## Run the model
Install the following pre-requisites:
* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```
* Navigate models directory and install model specific dependencies for the workload:
  ```bash
  # Navigate to the model zoo repo
  cd models
  # Install model specific dependencies:
  python -m pip install 'batchgenerators>=0.20.0' 'scipy==1.6.2' medpy pandas SimpleITK sklearn tensorboard
  ./quickstart/language_modeling/tensorflow/bert_large/training/gpu/setup.sh
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading the dataset.

```
# Set environment vars for the dataset and output path
export DATASET_DIR=<path the `results4` directory>
export OUTPUT_DIR=<directory where log files will be written>
export PROCESSED_DATASET_DIR=<path to `hdf5_seq_512` directory>
export Tile=2

# Run a quickstart script
./quickstart/language_modeling/pytorch/bert_large/training/gpu/<script_name>.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

