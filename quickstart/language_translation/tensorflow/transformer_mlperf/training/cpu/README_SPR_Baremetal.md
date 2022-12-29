<!--- 0. Title -->
# TensorFlow Transformer Language training

<!-- 10. Description -->

This document has instructions for running Transformer Language training on baremetal
using Intel-optimized TensorFlow.


<!-- 20. Environment setup on baremetal -->
## Setup on baremetal

* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```

* Install git, numactl and wget, if not installed already
  ```bash
  yum update -y && yum install -y git numactl wget
  ```

* Install Intel Tensorflow
  ```bash
  pip install intel-tensorflow==2.11.dev202242
  ```

* Install the keras version that works with the above tensorflow version:
  ```bash
  pip install keras-nightly==2.11.0.dev2022092907
  ```

* Note: For kernel version 5.16, AVX512_CORE_AMX is turned on by default. If the kernel version < 5.16 , please set the following environment variable for AMX environment: 
  ```bash
  DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  # To run VNNI, please set 
  DNNL_MAX_CPU_ISA=AVX512_CORE_BF16
  ```

* Clone [Intel Model Zoo repository](https://github.com/IntelAI/models)
  ```bash
  git clone https://github.com/IntelAI/models
  ```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training.sh` | Uses mpirun to execute 1 process per socket for the specified precision (fp32, bfloat32 or bfloat16). Logs for each instance are saved to the output directory. |

<!--- 30. Datasets -->
## Datasets

Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/transformer_data/README.md) to download and preprocess the WMT English-German dataset.
Set `DATASET_DIR` to point out to the location of the dataset directory.

<!--- 50. Baremetal -->

## Run the model

Set environment variables to
specify the dataset directory, precision to run, and
an output directory.
```
# Navigate to the model zoo directory
cd models

# Install pre-requisites for the model:
./quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/setup_spr.sh

# Set the required environment vars
export PRECISION=<specify the precision to run: fp32, bfloat16 or bfloat32>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run the quickstart scripts:
./quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/training.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

