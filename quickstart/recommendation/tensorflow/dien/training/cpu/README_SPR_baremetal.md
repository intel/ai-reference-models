<!--- 0. Title -->
# TensorFlow DIEN Training

<!-- 10. Description -->
## Description

This document has instructions for running DIEN training using
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
> Note: For `kernel version 5.16`, `AVX512_CORE_AMX` is turned on by default. If the `kernel version < 5.16` , please set the following environment variable for AMX environment: `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`. To run VNNI, please set `DNNL_MAX_CPU_ISA=AVX512_CORE_BF16`.

* Clone [Intel Model Zoo repository](https://github.com/IntelAI/models) if you haven't already cloned it.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`training.sh`](/quickstart/recommendation/tensorflow/dien/training/cpu/training.sh) | Runs training with a batch size of 128 for the specified precisions fp32, bfloat16 and bfloat32. |

<!--- 30. Datasets -->
## Datasets

Use [prepare_data.sh](https://github.com/alibaba/ai-matrix/blob/master/macro_benchmark/DIEN_TF2/prepare_data.sh) to get [a subset of the Amazon book reviews data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/) and process it.
Or download and extract the preprocessed data files directly:
```
wget https://zenodo.org/record/3463683/files/data.tar.gz
wget https://zenodo.org/record/3463683/files/data1.tar.gz
wget https://zenodo.org/record/3463683/files/data2.tar.gz

tar -jxvf data.tar.gz
mv data/* .
tar -jxvf data1.tar.gz
mv data1/* .
tar -jxvf data2.tar.gz
mv data2/* .
```
Set the `DATASET_DIR` to point to the directory with the dataset files when running <model name>.

## Run the model

After you've followed the instructions to [prepare the dataset](#datasets) using real data, set environment variables to
specify the path to the dataset directory, precision to run, and an output directory.

```
# Set the required environment vars
export PRECISION=<supported precisions are fp32, bfloat16, bfloat32>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
```

Navigate to the models directory to training.
```
cd models
```
### Run training:
```
./quickstart/recommendation/tensorflow/dien/training/cpu/training.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.
