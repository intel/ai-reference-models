<!--- 0. Title -->
# Transformer Language inference

<!-- 10. Description -->
## Description

This document has instructions for running Transformer Language inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/transformer_data/README.md) to download and preprocess the WMT English-German dataset.
Set `DATASET_DIR` to point out to the location of the dataset directory.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs inference (batch-size=1) to compute latency for the specified precision (int8, fp32, or bfloat16). |
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference (batch-size=1) to compute latency using 4 cores per instance for the specified precision (int8, fp32, or bfloat16). |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference with batch-size=448 for precisions (int8, fp32, bfloat16) to get throughput using 1 instance per socket. |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (int8, fp32, or bfloat16). |

## Run the model

Setup your environment using the instructions below,

* Follow the instructions to setup your bare metal environment on Linux system. Ensure that you have a clone of the [AI Reference Models Github repository](https://github.com/IntelAI/models).
  ```
  git clone https://github.com/IntelAI/models.git
  ```

* While using FP32 or BFloat16 precisions:

  Install [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/) on your system.

  While using Int8 precision:

  Install [intel-tensorflow==2.11.dev202242](https://pypi.org/project/intel-tensorflow/2.11.dev202242/) on your system.

After installing the prerequisites, download the pretrained model and set
the `PRETRAINED_MODEL` environment variable to the .pb file path:
```
# Int8 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/transformer_mlperf_int8.pb
export PRETRAINED_MODEL=$(pwd)/transformer_mlperf_int8.pb

# Int8 Pretrained model for OneDNN Graph (Only used when the plugin Intel Extension for Tensorflow is installed, as OneDNN Graph optimization is enabled by default at this point):
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_12_0/transformer_lt_itex_int8.pb
export PRETRAINED_MODEL=$(pwd)/transformer_lt_itex_int8.pb

# FP32 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/transformer_mlperf_fp32.pb
export PRETRAINED_MODEL=$(pwd)/transformer_mlperf_fp32.pb

# BFloat16 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_0/transformer_bf16_frozen_8_30.pb
export PRETRAINED_MODEL=$(pwd)/transformer_bf16_frozen_8_30.pb
```

Once that has been completed, ensure you have the required environment variables
set, and then run a quickstart script.

### Run on Linux
```
# cd to your AI Reference Models directory
cd models

# Set the required environment vars
export PRECISION=<specify the precision to run: int8, fp32, bfloat16>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pre-trained model>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run the quickstart scripts:
./quickstart/language_translation/tensorflow/transformer_mlperf/inference/cpu/<script_name>.sh
```

### Run on Windows
Using `cmd.exe`, run:
```
# cd to your AI Reference Models directory
cd models

# Set env vars
set DATASET_DIR=<path to the test dataset directory>
set PRECISION=<specify the precision to run: fp32, bfloat16>
set PRETRAINED_MODEL=<path to the downloaded pre-trained model>
set OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
set BATCH_SIZE=<customized batch size value>

# Run a quickstart script
bash quickstart\language_translation\tensorflow\transformer_mlperf\inference\cpu\<script name>.sh
```
> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables. 
As an example, if the pretrained model path on Windows is `D:\user\transformer_mlperf_fp32_pretrained_model\graph\fp32_graphdef.pb`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\transformer_mlperf_fp32_pretrained_model\graph\fp32_graphdef.pb
> /d/user/transformer_mlperf_fp32_pretrained_model/graph/fp32_graphdef.pb
>```
>Then, set the `PRETRAINED_MODEL` environment variable `set PRETRAINED_MODEL=/d/user/transformer_mlperf_fp32_pretrained_model/graph/fp32_graphdef.pb`.

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions for the available precisions [FP32](fp32/Advanced.md) [Int8](int8/Advanced.md) [BFloat16](bfloat16/Advanced.md) for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [Intel® Developer Catalog](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)
  workload container:<br />
  [https://www.intel.com/content/www/us/en/developer/articles/containers/transformer-lt-official-fp32-inference-tensorflow-container.html](https://www.intel.com/content/www/us/en/developer/articles/containers/transformer-lt-official-fp32-inference-tensorflow-container.html).
