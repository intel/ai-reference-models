<!--- 0. Title -->
# TensorFlow DIEN inference

<!-- 10. Description -->
## Description

This document has instructions for running DIEN inference using
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
> Note: For `kernel version 5.16`, `AVX512_CORE_AMX` is turned on by default. If the `kernel version < 5.16` , please set the following environment variable for AMX environment: `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`. To run VNNI, please set `DNNL_MAX_CPU_ISA=AVX512_CORE_BF16`.

* Clone [Intel Model Zoo repository](https://github.com/IntelAI/models) if you haven't already cloned it.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, bfloat16 or bfloat32) with a default`batch_size=16`. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, bfloat16 or bfloat32) with a default `batch_size=65536`. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, bfloat16 or bfloat32) with a default `batch_size=128`. |

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

## Download the pretrained model
Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```
# FP32, BFloat16 & BFloat32 Pretrained model for inference:
#TODO: To be publish externally
/tf_dataset/pre-trained-models/dien/dien_fp32_static_mklgrus.pb

# FP32, BFloat16 & BFloat32 Pretrained model for accuracy:
#TODO: To be publish externally
/tf_dataset/pre-trained-models/dien/dien_fp32_dynamic_mklgrus.pb
```

## Run the model

After you've followed the instructions to [download the pretrained model](#download-the-pretrained-model)
and [prepare the dataset](#datasets) using real data, set environment variables to
specify the path to the pretrained model, the dataset directory, precision to run, and an output directory.

You can change the defaut values for the batch size by setting `BATCH_SIZE` environemnt variable. Otherwise the default values in the [quick start scripts](#quick-start-scripts) will be used.

```
# Set the required environment vars
export PRECISION=<specify the precision to run>
export PRETRAINED_MODEL=<path to the downloaded pretrained model file>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
# Optional env vars
export BATCH_SIZE=<customized batch size value>
```

Navigate to the models directory to run any of the available benchmarks.
```
cd models
```
### Run real time inference (Latency):
```
./quickstart/recommendation/tensorflow/dien/inference/cpu/inference_realtime_multi_instance.sh
```

### Run inference (Throughput):
```
./quickstart/recommendation/tensorflow/dien/inference/cpu/inference_throughput_multi_instance.sh
```

### Run accuracy:
```
./quickstart/recommendation/tensorflow/dien/inference/cpu/accuracy.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.
