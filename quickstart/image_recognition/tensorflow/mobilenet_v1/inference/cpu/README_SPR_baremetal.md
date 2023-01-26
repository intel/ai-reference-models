<!--- 0. Title -->
# TensorFlow MobileNet V1 inference

<!-- 10. Description -->
## Description

This document has instructions for running MobileNet V1 inference using
Intel-optimized TensorFlow.

## Enviromnment setup

* Create a virtual environment `venv-tf` using `Python 3.8`:
```
pip install virtualenv
# use `whereis python` to find the `python3.8` path in the system and specify it. Please install `Python3.8` if not installed on your system.
virtualenv -p /usr/bin/python3.8 venv-tf
source venv-tf/bin/activate

# If git, numactl and wget were not installed, please install them using
yum update -y && yum install -y git numactl wget
```

* Install [Intel optimized TensorFlow](https://pypi.org/project/intel-tensorflow/2.11.dev202242/)
```
# Install Intel Optimized TensorFlow
pip install intel-tensorflow==2.11.dev202242
pip install keras-nightly==2.11.0.dev2022092907
```

* Clone [Intel Model Zoo repository](https://github.com/IntelAI/models) if you haven't already cloned it.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime_multi_instance.sh` | A multi-instance run that uses 4 cores per instance with `batch_size=1` for the specified precision (fp32, int8, bfloat16, bfloat32). Uses synthetic data if no DATASET_DIR is set|
| `inference_throughput_multi_instance.sh` | A multi-instance run that uses 4 cores per instance with `batch_size=448` for the specified precision (fp32, int8, bfloat16, bfloat32). Uses synthetic data if no DATASET_DIR is set |
| `accuracy.sh` | Measures the model accuracy (batch_size=100) for the specified precision (fp32, int8, bfloat16, bfloat32). |

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/tree/master/datasets/imagenet#imagenet-dataset-scripts).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running MobileNet V1(if needed).

## Download the pretrained model
Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```
# FP32, BFloat16 and BFloat32 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/mobilenetv1_fp32_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/mobilenetv1_fp32_pretrained_model.pb

# Int8 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/mobilenetv1_int8_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/mobilenetv1_int8_pretrained_model.pb
```

## Run the model

After you've followed the instructions to [download the pretrained model](#download-the-pretrained-model)
and [prepare the dataset](#datasets) using real data, set environment variables to
specify the path to the pretrained model, the dataset directory (if needed), precision to run, and an output directory.

The dataset is required for accuracy and optional for other inference scripts.
Optionally, you can change the defaut values for the batch size, by setting `BATCH_SIZE` environment variable. Otherwise the default values in the [quick start scripts](#quick-start-scripts) will be used.

>Note: 
For kernel version 5.16, AVX512_CORE_AMX is turned on by default. If the kernel version < 5.16 , please set the following environment variable for AMX environment: DNNL_MAX_CPU_ISA=AVX512_CORE_AMX. To run VNNI, please set DNNL_MAX_CPU_ISA=AVX512_CORE_BF16.


Navigate to the models directory to run any of the available benchmarks.
```
cd models

export PRECISION=<int8, bfloat16, bfloat32 and fp32 are supported>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pretrained model file for the used precision>
```
### Run real time inference (Latency):
```
./quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/inference_realtime_multi_instance.sh
```

### Run Throughput:
```
./quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/inference_throughput_multi_instance.sh
```

### Run accuracy:
```
# To test accuracy, also specify the dataset directory
export DATASET_DIR=<path to the dataset>

./quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/accuracy.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

