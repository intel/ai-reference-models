<!--- 0. Title -->
# BERT Large inference

<!-- 10. Description -->

This document has instructions for running BERT Large inference using
Intel-optimized TensorFlow with Intel® Data Center GPU Max Series.

<!--- 20. GPU Setup -->
## Software Requirements:
- Host machine has Intel® Data Center Max Series 1550 x4 OAM GPU
- Follow instructions to install GPU-compatible driver [647](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html)
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

### BERT Large Data
Download and unzip the BERT Large uncased (whole word masking) model from the [google bert repo](https://github.com/google-research/bert#pre-trained-models).
Then, download the Stanford Question Answering Dataset (SQuAD) dataset file `dev-v1.1.json` into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

```
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```
Set the `PRETRAINED_DIR` to point to that directory when running BERT Large inference using the SQuAD data.

Download the SQUAD directory and set the `SQUAD_DIR` environment variable to point where it was saved:
  ```
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
  wget https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
  ```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | This script runs Bert Large inference for precisions (fp16, bfloat16 and fp32). |


<!--- 50. Baremetal -->
## Run the model
* Download the frozen graph model file, and set the FROZEN_GRAPH environment variable to point to where it was saved:
  ```bash
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_0/bert_frozen_graph_fp32.pb
  ```

* Download the pretrained model directory and set the PRETRAINED_DIR environment variable to point where it was saved:
  ```bash
  wget  https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
  unzip wwm_uncased_L-24_H-1024_A-16.zip
  ```

* Download the SQUAD directory and set the SQUAD_DIR environment variable to point where it was saved:
  ```bash
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
  wget https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
  ```
* Clone the Model Zoo repository:
  ```bash
  git clone https://github.com/IntelAI/models.git
  ```

### Run the model on Baremetal
Navigate to the BERT Large inference directory, and set environment variables:
```
cd models
export OUTPUT_DIR=<path where output log files will be written>
export PRECISION=<provide either fp16,bfloat16 or fp32 precisions>
export FROZEN_GRAPH=<path to pretrained model file (*.pb)>
export PRETRAINED_DIR=<path to pretrained directory>
export SQUAD_DIR=<path to squad directory>
export NUM_OAM=<provide 4 for number of OAM Modules supported by the platform>
export INFERENCE_MODE=<provide accuracy,benchmark or profile mode>

# Run quickstart script:
./quickstart/language_modeling/tensorflow/bert_large/inference/gpu/<script name>.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

