<!--- 0. Title -->
# DIEN inference

<!-- 10. Description -->
## Description

This document has instructions for running DIEN inference using
Intel-optimized TensorFlow.

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
Set the `DATASET_DIR` to point to the directory with the dataset files when running DIEN.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs realtime inference using a default `batch_size=8` for the specified precision (fp32 or bfloat16) |
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, bfloat16 or bfloat32) with a default`batch_size=16`. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, bfloat16 or bfloat32) with a default `batch_size=65536`. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, bfloat16 or bfloat32) with a default `batch_size=128`. |

## Run the model
* Follow the instructions to setup your bare metal environment on either Linux or Windows systems. Ensure that you have a clone of the [AI Reference Models Github repository](https://github.com/IntelAI/models).
  ```
  git clone https://github.com/IntelAI/models.git
  ```

* Install [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/) on your system.

After finishing the setup above, download the pretrained model and set the
`PRETRAINED_MODEL` environment var to the path to the frozen graph.
If you run on Windows, please use a browser to download the pretrained model using the link below.

```
# FP32 Pretrained model
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/dien_fp32_static_rnn_graph.pb
export PRETRAINED_MODEL=$(pwd)/dien_fp32_static_rnn_graph.pb

# BFloat16 Pretrained model 
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/dien_bf16_pretrained_opt_model.pb
export PRETRAINED_MODEL=$(pwd)/dien_bf16_pretrained_opt_model.pb

```

## Run on Linux
```
# cd to your AI Reference Models directory
cd models

# Set env vars
export DATASET_DIR=<path to the DIEN dataset>
export PRECISION=<set the precision to "fp32" or "bfloat16">
export OUTPUT_DIR=<path to the directory where log files will be written>
export PRETRAINED_MODEL=<path to the frozen graph downloaded above>

# Run a quickstart script
./quickstart/recommendation/tensorflow/dien/inference/cpu/<script name>.sh
```

## Run on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).

Using Windows CMD.exe, run:
```
cd models

# Set environment variables
set DATASET_DIR=<path to the DIEN dataset>
set PRECISION=<set the precision to "fp32" or "bfloat16">
set OUTPUT_DIR=<path to the directory where log files will be written>
set PRETRAINED_MODEL=<path to the frozen graph downloaded above>

# Run a quick start script for inference or accuracy
bash quickstart\recommendation\tensorflow\dien\inference\cpu\inference.sh
```

> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables.
As an example, if the pretrained model path on Windows is `D:\user\dien_fp32_static_rnn_graph.pb`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\dien_fp32_static_rnn_graph.pb
> /d/user/dien_fp32_static_rnn_graph.pb
>```
>Then, set the `PRETRAINED_MODEL` environment variable `set PRETRAINED_MODEL=/d/user/dien_fp32_static_rnn_graph.pb`.

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
