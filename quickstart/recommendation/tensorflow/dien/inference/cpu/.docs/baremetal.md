<!--- 50. AI Kit -->
## Run the model
* Follow the instructions to setup your bare metal environment on either Linux or Windows systems. Ensure that you have a clone of the [Model Zoo Github repository](https://github.com/IntelAI/models).
  ```
  git clone https://github.com/IntelAI/models.git
  ```

* Install [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/) on your system.

After finishing the setup above, download the pretrained model and set the
`PRETRAINED_MODEL` environment var to the path to the frozen graph.
If you run on Windows, please use a browser to download the pretrained model using the link below.

```
# FP32 and BFloat16 Pretrained model
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/dien_fp32_static_rnn_graph.pb
export PRETRAINED_MODEL=$(pwd)/dien_fp32_static_rnn_graph.pb

# BFloat16 Pretrained model 
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/dien_bf16_pretrained_opt_model.pb
export PRETRAINED_MODEL=$(pwd)/dien_bf16_pretrained_opt_model.pb

```

## Run on Linux
```
# cd to your model zoo directory
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
