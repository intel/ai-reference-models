<!--- 50. Baremetal -->
## Pre-Trained Model

Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```bash
# INT8:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/3dunet_dynamic_ndhwc.pb

# FP32 and BFloat32:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/3dunet_dynamic_ndhwc.pb

# BFloat16:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/3dunet_dynamic_ndhwc.pb
```

## Run the model

Set environment variables to
specify pretrained model, the dataset directory, precision to run, and an output directory. 
```
# Navigate to the model zoo repository
cd models

# Install pre-requisites for the model:
pip install -r benchmarks/image_segmentation/tensorflow/3d_unet_mlperf/requirements.txt

# Set the required environment vars
export PRECISION=<specify the precision to run: int8, fp32, bfloat16 and bfloat32>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pre-trained model>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Dataset is only required for running accuracy script, Inference scripts don't require 'DATASET_DIR' to be set:
export DATASET_DIR=<path to the dataset>

# Run the script:
./quickstart/image_segmentation/tensorflow/3d_unet_mlperf/inference/cpu/<script_name>.sh
```

