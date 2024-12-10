<!--- 0. Title -->
# TensorFlow BERT Large Pretraining

<!-- 10. Description -->

This document has instructions for running BERT Large Pretraining on baremetal
using Intel-optimized TensorFlow.

<!-- 20. Environment setup on baremetal -->
## Setup on baremetal

* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```

* Install Intel Tensorflow
  ```bash
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
| `pretraining.sh` | Uses mpirun to execute 1 process per socket for BERT Large pretraining with the specified precision (fp32, bfloat16 and bfloat32). Logs for each instance are saved to the output directory. |

<!--- 30. Datasets -->
## Datasets

### SQuAD data
Download and unzip the BERT Large uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Set the `DATASET_DIR` to point to this directory when running BERT Large.
```
mkdir -p $DATASET_DIR && cd $DATASET_DIR
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```

Follow [instructions to generate BERT pre-training dataset](https://github.com/IntelAI/models/blob/bert-lamb-pretraining-tf-2.2/quickstart/language_modeling/tensorflow/bert_large/training/bfloat16/HowToGenerateBERTPretrainingDataset.txt)
in TensorFlow record file format. The output TensorFlow record files are expected to be located in the dataset directory `${DATASET_DIR}/tf_records`. An example for the TF record file path should be
`${DATASET_DIR}/tf_records/part-00430-of-00500`.

<!--- 50. Baremetal -->

## Download checkpoints:
```bash
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
unzip bert_large_checkpoints.zip
export CHECKPOINT_DIR=$(pwd)/bert_large_checkpoints
```

## Run the model

Set environment variables to
specify the dataset directory, precision to run, and
an output directory.
```
# Navigate to the container package directory
cd models

# Install pre-requisites for the model:
./models_v2/tensorflow/bert_large/training/cpu/setup.sh

# Set the required environment vars
export PRECISION=<specify the precision to run:fp32, bfloat16 and bfloat32>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export CHECKPOINT_DIR=<path to the downloaded checkpoints folder>

# Run the container with pretraining.sh quickstart script
./models_v2/tensorflow/bert_large/training/cpu/pretraining.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

