<!--- 50. Baremetal -->
## Pre-Trained Model

Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```bash
# INT8:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/transformer_mlperf_int8.pb

# FP32:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/transformer_mlperf_fp32.pb

# BFloat16:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/transformer_mlperf_bf16.pb
```

## Run the model

Set environment variables to
specify the dataset directory, pretrained model path, precision to run, and
an output directory.
```
# Navigate to the container package directory
cd models

# Install pre-requisites for the model:
./quickstart/language_translation/tensorflow/transformer_mlperf/inference/cpu/setup_spr.sh

# Set the required environment vars
export PRECISION=<specify the precision to run: int8, fp32, bfloat16 or bfloat32>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pre-trained model>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run the quickstart scripts:
./quickstart/language_translation/tensorflow/transformer_mlperf/inference/cpu/<script_name>.sh
