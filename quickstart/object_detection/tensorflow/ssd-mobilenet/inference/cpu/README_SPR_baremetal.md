<!--- 0. Title -->
# TensorFlow SSD-MobileNet inference

<!-- 10. Description -->
## Description

This document has instructions for running SSD-MobileNet inference using
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

* After cloning Model Zoo repository, install model specific dependencies
```
pip install -r requirements.txt
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`accuracy.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/accuracy.sh) | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, bfloat16). |
| [`inference_throughput_multi_instance.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference_throughput_multi_instance.sh) | A multi-instance run that uses all the cores for each socket for each instance with a batch size of 448 and synthetic data. |
| [`inference_realtime_multi_instance.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference_realtime_multi_instance.sh) | A multi-instance run that uses 4 cores per instance with a batch size of 1. Uses synthetic data if no `DATASET_DIR` is set. |

<!--- 30. Datasets -->
## Datasets

The [COCO validation dataset](http://cocodataset.org) is used in these
SSD-Mobilenet quickstart scripts. The accuracy quickstart script require the dataset to be converted into the TF records format.
See the [COCO dataset](https://github.com/IntelAI/models/tree/master/datasets/coco) for instructions on
downloading and preprocessing the COCO validation dataset.

Set the `DATASET_DIR` to point to the dataset directory that contains the TF records file `coco_val.record` when running SSD-MobileNet accuracy script.

## Download the pretrained model
Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```
# FP32, BFloat16 and BFloat32 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb
export PRETRAINED_MODEL=$(pwd)/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb

# Int8 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssdmobilenet_int8_pretrained_model_combinedNMS_s8.pb
export PRETRAINED_MODEL=$(pwd)/ssdmobilenet_int8_pretrained_model_combinedNMS_s8.pb

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

export PRECISION=<int8, bfloat16, bfloat32, fp32 are supported>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pretrained model for the used precision>

# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>
```
### Run real time inference (Latency):
```
./quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference_realtime_multi_instance.sh
```

### Run inference (Throughput):
```
./quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference_throughput_multi_instance.sh
```

### Run accuracy:
```
# To test accuracy, also specify the dataset directory
export DATASET_DIR=<path to the dataset>

./quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/accuracy.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.
