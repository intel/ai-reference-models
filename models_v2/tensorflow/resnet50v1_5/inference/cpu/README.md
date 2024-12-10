<!--- 0. Title -->
# TensorFlow ResNet50 v1.5 Inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 v1.5 inference using
Intel-optimized TensorFlow.

## Enviromnment setup

* Create a virtual environment `venv-tf`:
```
python -m venv venv-tf
source venv-tf/bin/activate
```

* Install [Intel optimized TensorFlow](https://pypi.org/project/intel-tensorflow/)
```
# Install Intel Optimized TensorFlow
pip install intel-tensorflow
```

* Clone [Intel AI Reference Models repository](https://github.com/IntelAI/models) if you haven't already cloned it.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, int8, bfloat16, bfloat32) with 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_realtime_weightsharing.sh` | Runs multi instance realtime inference with weight sharing for the specified precision (int8 or bfloat16) with 1500 steps and 100 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, int8, bfloat16, bfloat32) with 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8, bfloat16, bfloat32). |


<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/tree/master/datasets/imagenet#imagenet-dataset-scripts).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5 (if needed).

## Download the pretrained model
Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```
# BFloat16 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_8/bf16_resnet50_v1.pb
export PRETRAINED_MODEL=$(pwd)/bf16_resnet50_v1.pb

# FP32 and BFloat32 Pretrained model:
wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
export PRETRAINED_MODEL=$(pwd)/resnet50_v1.pb

# Int8 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_8/bias_resnet50.pb
export PRETRAINED_MODEL=$(pwd)/bias_resnet50.pb
```

## Run the model

After you've followed the instructions to [download the pretrained model](#download-the-pretrained-model)
and [prepare the dataset](#datasets) using real data, set environment variables to
specify the path to the pretrained model, the dataset directory (if needed), precision to run, and an output directory.

The dataset is required for accuracy and optional for other inference scripts.
Optionally, you can change the defaut values for the batch size, warmup steps and steps by setting `BATCH_SIZE`, `WARMUP_STEPS`, and `STEPS`. Otherwise the default values in the [quick start scripts](#quick-start-scripts) will be used.

>Note:
For kernel version 5.16, AVX512_CORE_AMX is turned on by default. If the kernel version < 5.16 , please set the following environment variable for AMX environment: DNNL_MAX_CPU_ISA=AVX512_CORE_AMX. To run VNNI, please set DNNL_MAX_CPU_ISA=AVX512_CORE_BF16.

Navigate to the models directory to run any of the available benchmarks.
```
cd models

# Set the required environment vars
export PRECISION=<Supported precisions are fp32, int8, bfloat16 and bfloat32>
export PRETRAINED_MODEL=<path to the downloaded pretrained model file>
export OUTPUT_DIR=<directory where log files will be written>

# Optional env vars
export BATCH_SIZE=<customized batch size value, otherwise it will run with the default value>
export WARMUP_STEPS=<customized warm up steps value, otherwise it will run with the default value>
export STEPS=<customized steps value, otherwise it will run with the default value>
export OMP_NUM_THREADS=<customized value for omp_num_threads, otherwise it will run with the default value>
export CORES_PER_INSTANCE=<customized value for cores_per_instance, otherwise it will run with the default value>
```
### Run real time inference (Latency):
```
./models_v2/tensorflow/resnet50v1_5/inference/cpu/inference_realtime_multi_instance.sh
```

### Run inference (Throughput):
```
./models_v2/tensorflow/resnet50v1_5/inference/cpu/inference_throughput_multi_instance.sh
```

### Run real time inference (Latency with weight sharing enabled):
```
# only int8 and bfloat16 precisions are supported for weight sharing
export PRECISION=<int8 or bfloat16 are supported>

./models_v2/tensorflow/resnet50v1_5/inference/cpu/inference_realtime_weightsharing.sh
```

### Run accuracy:
```
# To test accuracy, also specify the dataset directory
export DATASET_DIR=<path to the dataset>

./models_v2/tensorflow/resnet50v1_5/inference/cpu/accuracy.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.
