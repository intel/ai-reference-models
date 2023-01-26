<!--- 0. Title -->
# TensorFlow MLPerf 3D U-Net inference

<!-- 10. Description -->
## Description

This document has instructions for running MLPerf 3D U-Net inference on baremetal using
Intel-optimized TensorFlow.

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
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (int8, fp32, bfloat32 or bfloat16) with 100 steps and 50 warmup steps to compute latency. Dummy data is used for performance evaluation. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (int8, fp32, bfloat32 or bfloat16) with 100 steps and 50 warmup steps to compute throughput. Dummy data is used for performance evaluation. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (int8, fp32, bfloat32 or bfloat16). |

<!--- 30. Datasets -->
## Datasets

Download [Brats 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) separately and unzip the dataset.

Set the `DATASET_DIR` to point to the directory that contains the dataset files when running MLPerf 3D U-Net accuracy script.

<!--- 50. Baremetal -->
## Pre-Trained Model

Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```bash
# INT8:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/3dunet_new_int8_bf16.pb

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

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

