<!--- 0. Title -->
# TensorFlow Transformer Language training

<!-- 10. Description -->

This document has instructions for running Transformer Language training on baremetal
using Intel-optimized TensorFlow.

<!-- 20. Environment setup on baremetal -->
## Setup on baremetal

* Create a virtual environment `venv-tf`:
```
python -m venv venv-tf
source venv-tf/bin/activate
```

* Install [Intel optimized TensorFlow](https://pypi.org/project/intel-tensorflow/)
```
# Install Intel Optimized TensorFlow
pip install intel-tensorflow
```

* Note: For kernel version 5.16, AVX512_CORE_AMX is turned on by default. If the kernel version < 5.16 , please set the following environment variable for AMX environment: 
  ```bash
  DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  # To run VNNI, please set 
  DNNL_MAX_CPU_ISA=AVX512_CORE_BF16
  ```

* Clone [Intel AI Reference Models repository](https://github.com/IntelAI/models)
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
# Navigate to the models directory
cd models

# Install pre-requisites for the model:
./quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/setup.sh

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

