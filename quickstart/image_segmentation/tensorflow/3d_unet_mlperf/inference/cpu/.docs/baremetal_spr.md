<!--- 50. Baremetal -->
## Pre-Trained Model

Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.

Paths to the pb files relative to tf_dataset:
* INT8:
  BERT-Large-squad_int8 = /tf_dataset/pre-trained-models/3DUNet/int8/3dunet_new_int8_bf16.pb

* FP32 and BFloat32:
  BERT-Large-squad_fp32 = /tf_dataset/pre-trained-models/3DUNet/fp32/3dunet_dynamic_ndhwc.pb

* BFloat16:
  BERT-Large-squad_bfloat16 = /tf_dataset/pre-trained-models/3DUNet/bfloat16/3dunet_dynamic_ndhwc.pb

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run MLPerf 3D U-Net inference in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. 
```
# Navigate to the model zoo repository
cd models

# Set the required environment vars
export PRECISION=<specify the precision to run: int8, fp32, bfloat16 and bfloat32>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pre-trained model>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run the script:
./quickstart/image_segmentation/tensorflow/3d_unet_mlperf/inference/cpu/<script_name>.sh
```

