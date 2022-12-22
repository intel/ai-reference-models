<!--- 0. Title -->
# TensorFlow ResNet50 v1.5 Training

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 v1.5 training using
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

>Note: For kernel version 5.16, AVX512_CORE_AMX is turned on by default. If the kernel version < 5.16 , please set the following environment variable for AMX environment: DNNL_MAX_CPU_ISA=AVX512_CORE_AMX. To run VNNI, please set DNNL_MAX_CPU_ISA=AVX512_CORE_BF16.

* Clone [Intel Model Zoo repository](https://github.com/IntelAI/models) if you haven't already cloned it.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`multi_instance_training.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/multi_instance_training.sh) | Uses mpirun to execute 1 processes with 1 process per socket with a batch size of 1024 for the specified precision (fp32 or bfloat16 or bfloat32). Checkpoint files and logs for each instance are saved to the output directory.|

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/tree/master/datasets/imagenet#imagenet-dataset-scripts).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5 (if needed).

## Run the model

After finishing the setup above, set environment variables for the path to your DATASET_DIR for ImageNet and an OUTPUT_DIR where log files and checkpoints will be written. Navigate to your model zoo directory and then run a quickstart script.

# cd to your model zoo directory

```
# Set the required environment vars
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>
# Optional env vars
export BATCH_SIZE=<customized batch size value>
```

Navigate to the models directory to run any of the available benchmarks.
```
cd models

./quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/multi_instance_training.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

