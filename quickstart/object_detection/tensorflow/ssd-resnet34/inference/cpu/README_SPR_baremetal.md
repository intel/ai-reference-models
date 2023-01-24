<!--- 0. Title -->
# TensorFlow SSD-ResNet34 inference

<!-- 10. Description -->
## Description

This document has instructions for running SSD-ResNet34 inference using
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
./quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/setup_spr.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [multi_instance_online_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/multi_instance_online_inference_1200.sh) | Runs multi instance realtime inference (batch-size=1) using 4 cores per instance for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |
| [multi_instance_batch_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/fp32/multi_instance_batch_inference_1200.sh) | Runs multi instance batch inference (batch-size=16) using 1 instance per socket for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |
| [accuracy_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/accuracy_1200.sh) | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bfloat16) with an input size of 1200x1200. |

<!--- 30. Datasets -->
## Datasets

The SSD-ResNet34 accuracy script `accuracy.sh` uses the
[COCO validation dataset](http://cocodataset.org) in the TF records
format. See the [COCO dataset document](https://github.com/IntelAI/models/tree/master/datasets/coco) for
instructions on downloading and preprocessing the COCO validation dataset.
The inference scripts use synthetic data, so no dataset is required.

After the script to convert the raw images to the TF records file completes, rename the tf_records file:
```
mv ${OUTPUT_DIR}/coco_val.record ${OUTPUT_DIR}/validation-00000-of-00001
```
Set the `DATASET_DIR` to the folder that has the `validation-00000-of-00001`
file when running the accuracy test. Note that the inference performance
test uses synthetic dataset.

The [TensorFlow models](https://github.com/tensorflow/models) and
[benchmarks](https://github.com/tensorflow/benchmarks) repos are used by
SSD-ResNet34 inference. Clone those at the git SHAs specified
below and set the `TF_MODELS_DIR` environment variable to point to the
directory where the models repo was cloned.

```
git clone --single-branch https://github.com/tensorflow/models.git tf_models
git clone --single-branch https://github.com/tensorflow/benchmarks.git ssd-resnet-benchmarks
cd tf_models
export TF_MODELS_DIR=$(pwd)
git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
cd ../ssd-resnet-benchmarks
git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
cd ..
```

## Download the pretrained model
Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```
# FP32, BFloat16 and BFloat32 Pretrained model:
wget  https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/ssd_resnet34_fp32_1200x1200_pretrained_model.pb

# Int8 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/ssd_resnet34_int8_1200x1200_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/ssd_resnet34_int8_1200x1200_pretrained_model.pb
```

## Run the model

After you've followed the instructions to [download the pretrained model](#download-the-pretrained-model)
and [prepare the dataset](#datasets) using real data, set environment variables to
specify the path to the pretrained model, the dataset directory (if needed), precision to run, and an output directory.

The dataset is required for accuracy and optional for other inference scripts.
Optionally, you can change the defaut value for the batch size by setting `BATCH_SIZE`. Otherwise the default value in the [quick start scripts](#quick-start-scripts) will be used.

>Note:
For kernel version 5.16, AVX512_CORE_AMX is turned on by default. If the kernel version < 5.16 , please set the following environment variable for AMX environment: DNNL_MAX_CPU_ISA=AVX512_CORE_AMX. To run VNNI, please set DNNL_MAX_CPU_ISA=AVX512_CORE_BF16.

Navigate to the models directory to run any of the available benchmarks.
```
cd models

# Set the required environment vars
export DATASET_DIR=<directory with the validation-*-of-* files (for accuracy testing only)>
export TF_MODELS_DIR=<path to the TensorFlow Models repo>
export PRECISION=<specify the precision to int8 or fp32 or bfloat16 or bfloat32>
export PRETRAINED_MODEL=<path to the downloaded pretrained model file>
export OUTPUT_DIR=<directory where log files will be written>

# Optional env vars
export BATCH_SIZE=<customized batch size value>
```
### Run real time inference (Latency):
```
./quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/multi_instance_online_inference_1200.sh
```

### Run inference (Throughput):
```
./quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/multi_instance_batch_inference_1200.sh
```

### Run accuracy:
```
./quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/accuracy_1200.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.
