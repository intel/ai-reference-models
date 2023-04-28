<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below,

* Follow the instructions to setup your bare metal environment on Linux system. Ensure that you have a clone of the [Model Zoo Github repository](https://github.com/IntelAI/models).
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

# FP32 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/transformer_mlperf_fp32.pb
export PRETRAINED_MODEL=$(pwd)/transformer_mlperf_fp32.pb

# BFloat16 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/transformer_mlperf_bf16.pb
export PRETRAINED_MODEL=$(pwd)/transformer_mlperf_bfloat16.pb
```

Once that has been completed, ensure you have the required environment variables
set, and then run a quickstart script.

### Run on Linux
```
# cd to your model zoo directory
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
# cd to your model zoo directory
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
