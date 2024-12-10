<!--- 0. Title -->
# TensorFlow ResNet50 v1.5 Training

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 v1.5 training using
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

>Note: For kernel version 5.16, AVX512_CORE_AMX is turned on by default. If the kernel version < 5.16 , please set the following environment variable for AMX environment: DNNL_MAX_CPU_ISA=AVX512_CORE_AMX. To run VNNI, please set DNNL_MAX_CPU_ISA=AVX512_CORE_BF16.
* Clone [Intel AI Reference Models repository](https://github.com/IntelAI/models) if you haven't already cloned it.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`multi_instance_training.sh`](/models_v2/tensorflow/resnet50v1_5/training/cpu/multi_instance_training.sh) | Uses mpirun to execute 1 processes with 1 process per socket with a batch size of 1024 for the specified precision (fp32 or bfloat16 or bfloat32 or fp16). Checkpoint files and logs for each instance are saved to the output directory.|

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/tree/master/datasets/imagenet#imagenet-dataset-scripts).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5 (if needed).

## Run the model

After finishing the setup above, set environment variables for the path to your DATASET_DIR for ImageNet and an OUTPUT_DIR where log files and checkpoints will be written. Navigate to your AI Reference Models  directory and then run a quickstart script.

# cd to your AI Reference Models  directory

```
# Set the required environment vars
export PRECISION=<specify the precision to run: fp32 or bfloat16 or bfloat32 or fp16>
export OUTPUT_DIR=<directory where log files will be written>
export DATASET_DIR=<set path to the dataset directory>

# Optional env vars
export BATCH_SIZE=<set batch size value else it will run with default value>
```

Navigate to the models directory to run any of the available benchmarks.
```
cd models

./models_v2/tensorflow/resnet50v1_5/training/cpu/multi_instance_training.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

