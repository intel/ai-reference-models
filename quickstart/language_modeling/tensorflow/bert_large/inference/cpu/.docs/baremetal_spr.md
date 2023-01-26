<!--- 50. Baremetal -->
## Pre-Trained Model

Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```bash
# INT8:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/per_channel_opt_int8_bf16_bert.pb

#FP32 and BFloat32:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/fp32_bert_squad.pb

#BFloat16:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/optimized_bf16_bert.pb
```

## Download checkpoints:
```bash
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
unzip bert_large_checkpoints.zip
export CHECKPOINT_DIR=$(pwd)/bert_large_checkpoints
```

## Run the model

Set environment variables to specify the dataset directory, precision to run, path to pretrained files and an output directory.
```
# Navigate to the models directory
cd models

# Set the required environment vars
export PRECISION=<specify the precision to run: int8, fp32 , bfloat32 and bfloat16>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pre-trained model>
export CHECKPOINT_DIR=<path to the downloaded checkpoints folder>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

Run the script:
./quickstart/language_modeling/tensorflow/bert_large/inference/cpu/<script_name.sh>
```
