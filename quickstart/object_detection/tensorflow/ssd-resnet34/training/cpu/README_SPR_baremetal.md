<!--- 0. Title -->
# TensorFlow SSD-ResNet34 Training

<!-- 10. Description -->
## Description

This document has instructions for running SSD-ResNet34 training using
Intel-optimized TensorFlow.

## Enviromnment setup

* Create a virtual environment `venv-tf` using `Python 3.8`:
```
pip install virtualenv
# use `whereis python` to find the `python3.8` path in the system and specify it. Please install `Python3.8` if not installed on your system.
virtualenv -p /usr/bin/python3.8 venv-tf
source venv-tf/bin/activate
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
| `training.sh` | Uses mpirun to execute 1 processes with 1 process per socket with a batch size of 896 for the specified precision (fp32 or bfloat16 or bfloat32). Logs for each instance are saved to the output directory. |

<!--- 30. Datasets -->
## Datasets

SSD-ResNet34 training uses the COCO training dataset. Use the [instructions](https://github.com/IntelAI/models/tree/master/datasets/coco/README_train.md) to download and preprocess the dataset.

## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit</th>
    <th>Setup without AI Kit</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Kit you will need:</p>
      <ul>
        <li>git
        <li>numactl
        <li>contextlib2
        <li>cpio
        <li>Cython
        <li>horovod
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>numpy>=1.17.4
        <li>opencv
        <li>openmpi
        <li>openssh
        <li>pillow>=9.3.0
        <li>protoc
        <li>pycocotools
        <li>tensorflow-addons==0.11.0
        <li>Activate the tensorflow 2.5.0 conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li>git
        <li>numactl
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>contextlib2
        <li>cpio
        <li>Cython
        <li>horovod==0.25.0
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>opencv
        <li>openmpi-4.1.0
        <li>openssh
        <li>pillow>=9.3.0
        <li>protoc
        <li>pycocotools
        <li>tensorflow-addons==0.18.0
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

For more information on the dependencies, see the
[installation instructions](https://github.com/tensorflow/models/blob/8110bb64ca63c48d0caee9d565e5b4274db2220a/research/object_detection/g3doc/installation.md#installation)
for object detection models at the
[TensorFlow Model Garden](https://github.com/tensorflow/models) repository.

Running SSD-ResNet34 training uses code from the
[TensorFlow Model Garden](https://github.com/tensorflow/models).
Clone the  repo at the commit specified below, and set the `TF_MODELS_DIR` environment
variable to point to that directory.
```
# Clone the tensorflow/models repo at the specified commit.
# Please note that required commit for this section is different from the one used for dataset preparation.
git clone https://github.com/tensorflow/models.git tf_models
cd tf_models
export TF_MODELS_DIR=$(pwd)
git checkout 8110bb64ca63c48d0caee9d565e5b4274db2220a
cd ..
```

After finishing the setup above, set environment variables and run training script on Linux. Navigate to your model zoo directory and then run a quickstart script.


```
# cd to your model zoo directory
cd models

# Set the required environment vars
export TF_MODELS_DIR=<path to your clone of the TensorFlow models repo>
export PRECISION=<set precision to fp32 or bfloat16 or bfloat32>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<path to the directory where log and checkpoint files will be written>

# Optional env vars
export MPI_NUM_PROCESSES=<number of MPI processes (optional, defaults to 1)
export BATCH_SIZE=<customized batch size value>

./quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/training.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

