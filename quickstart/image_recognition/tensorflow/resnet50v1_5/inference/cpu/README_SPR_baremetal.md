<!--- 0. Title -->
# TensorFlow ResNet50 v1.5 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 v1.5 inference using
Intel-optimized TensorFlow.

## Enviromnment setup

* Create a virtual environment `venv-tf` using `Python 3.8`:
```
pip install virtualenv
# use `whereis python` to find the `python3.8` path in the system and specify it. Please install `Python3.8` if not installed on your system.
virtualenv -p /usr/bin/python3.8 venv-tf
source venv-tf/bin/activate
```

* Install [Intel optimized TensorFlow](https://pypi.org/project/intel-tensorflow-avx512/2.10.dev202230/)
```
# Install Intel Optimized TensorFlow
pip install intel-tensorflow-avx512==2.10.dev202230
```

* Clone [Intel Model Zoo repository](https://github.com/IntelAI/models) if you haven't already cloned it.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`inference_realtime.sh`](inference_realtime.sh) | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, int8 or bfloat16). If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| [`inference_realtime_weightsharing.sh`](inference_realtime_weightsharing.sh) | Runs multi instance realtime inference with weight sharing for the specified precision (int8 or bfloat16). If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| [`inference_throughput.sh`](inference_throughput.sh) | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, int8 or bfloat16). If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| [`accuracy.sh`](accuracy.sh) | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bfloat16). |

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/tree/master/datasets/imagenet#imagenet-dataset-scripts).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5 (if needed).

## Download the pretrained model
Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```
# FP32 Pretrained model:
wget https://zenodo.org/record/2535873/files/resnet50_v1.pb

# Int8 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_8/bias_resnet50.pb

# BFloat16 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_8/bf16_resnet50_v1.pb
```

## Run the model

After you've followed the instructions to [download the pretrained model](#download-the-pretrained-model)
and [prepare the dataset](#datasets) using real data, set environment variables to
specify the path to the pretrained model, the dataset directory (if needed), precision to run, and an output directory.

The dataset is required for accuracy and optional for other inference scripts.
Optionally, you can change the defaut values for the batch size, warmup steps and steps by setting `BATCH_SIZE`, `WARMUP_STEPS`, and `STEPS`. Otherwise the default values in the [quick start scripts](#quick-start-scripts) will be used.

By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Set the required environment vars
export PRECISION=<specify the precision to run>
export PRETRAINED_MODEL=<path to the downloaded pretrained model file>
export OUTPUT_DIR=<directory where log files will be written>
# Optional env vars
export BATCH_SIZE=<customized batch size value>
export WARMUP_STEPS=<customized warm up steps value>
export STEPS=<customized steps value>
```

>Note: 
>* Use AVX3 instruction: If `kernel version >= 5.15` please set this environment variable, otherwise you can ignore it. 
>```
>export ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16
>```
>* Use AMX instruction: If you are using `official linux kernel` and `version > 5.15`, you can ignore this environment variable. If you are using `Intel-next kernel`, to use AMX instrucstion make sure `kernel version > 5.9` and set the environment variable.
>```
>export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX
>```
>For `Intel next kernel > 5.15`, you can ignore this environment variable.


Navigate to the models directory to run any of the available benchmarks.
```
cd models
```
### Run real time fp32 inference (Latency):
```
export PRECISION="fp32"
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded fp32 pretrained model file>

./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/inference_realtime.sh
```

### Run real time fp32 inference (Latency with weight sharing enabled):
```
export PRECISION=<int8 or bfloat16 are supported>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pretrained model file for the used precision>

./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/inference_realtime_weightsharing.sh
```

### Run inference (Throughput):
```
export PRECISION=<int8, bfloat16 or fp32 are supported>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pretrained model file for the used precision>

./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/inference_throughput.sh
```

### Run accuracy:
```
export PRECISION=<int8, bfloat16 or fp32 are supported>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pretrained model file for the used precision>
# To test accuracy, also specify the dataset directory
export DATASET_DIR=<path to the dataset>

./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/accuracy.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

