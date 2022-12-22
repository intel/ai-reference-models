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
./quickstart/language_modeling/tensorflow/bert_large/training/cpu/setup_spr.sh

# Set the required environment vars
export PRECISION=<specify the precision to run:fp32, bfloat16 and bfloat32>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export CHECKPOINT_DIR=<path to the downloaded checkpoints folder>

# Run the container with pretraining.sh quickstart script
./quickstart/language_modeling/tensorflow/bert_large/training/cpu/pretraining.sh
```
