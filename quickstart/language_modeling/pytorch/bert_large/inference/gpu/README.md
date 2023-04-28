<!--- 0. Title -->
# BERT Large inference for Intel(R) Data Center GPU Max Series

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large inference using
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

### SQuAD dataset

Download the [SQuAD 1.0 dataset](https://github.com/huggingface/transformers/tree/v4.0.0/examples/question-answering#fine-tuning-bert-on-squad10).
Set the `DATASET_DIR` to point to the directory where the files are located before
running the BERT quickstart scripts. Your dataset directory should look something
like this:
```
<DATASET_DIR>/
├── dev-v1.1.json
├── evaluate-v1.1.py
└── train-v1.1.json
```
The setup assumes the dataset is downloaded to the current directory. 

## Pre-trained Model

Download the `config.json` and fine tuned model from huggingface and set the `BERT_WEIGHT` environment variable to point to the directory that has both files:

```
mkdir bert_squad_model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
BERT_WEIGHT=$(pwd)/bert_squad_model
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp16_inference_plain_format.sh](fp16_inference_plain_format.sh) | Runs BERT large FP16 inference (plain format) using the SQuAD dataset |

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
  python -m pip install -r models/language_modeling/pytorch/bert_large/inference/gpu/requirements.txt
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading the SQuAD dataset.

```
# Set environment vars for the dataset and an output directory
export DATASET_DIR=<path the dataset directory>
export OUTPUT_DIR=<directory where log files will be written>
export BERT_WEIGHT=<directory where BERT weight files will be downloaded>
export Tile=2

# Run a quickstart script
./quickstart/language_modeling/pytorch/bert_large/inference/gpu/fp16_inference_plain_format.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

