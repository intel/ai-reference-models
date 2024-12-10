<!--- 0. Title -->
# Wide & Deep inference

<!-- 10. Description -->

This document has instructions for running Wide & Deep inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Dataset
Download and preprocess the [income census data](https://archive.ics.uci.edu/ml/datasets/Census+Income) by running
following python script, which is a standalone version of [census_dataset.py](https://github.com/tensorflow/models/blob/v2.2.0/official/r1/wide_deep/census_dataset.py)
Please note that below program requires `requests` module to be installed. You can install it using `pip install requests`.
Dataset will be downloaded in directory provided using `--data_dir`. If you are behind corporate proxy, then you can provide proxy URLs
using `--http_proxy` and `--https_proxy` arguments.
```
git clone https://github.com/IntelAI/models.git
cd models
python ./benchmarks/recommendation/tensorflow/wide_deep/inference/fp32/data_download.py --data_dir /home/<user>/widedeep_dataset
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`inference_online.sh`](/models_v2/tensorflow/wide_deep/inference/cpu/inference_online.sh) | Runs wide & deep model inference online mode (batch size = 1)|
| [`inference_batch.sh`](/models_v2/tensorflow/wide_deep/inference/cpu/inference_batch.sh) | Runs wide & deep model inference in batch mode (batch size = 1024)|

### Software requirements:
Install [tensorflow](https://pypi.org/project/tf_nightly/)

### Pretrained Model
Download and extract the pretrained
model. If you run on Windows, please use a browser to download the pretrained model using the link below.
Set the directory path to the `PRETRAINED_MODEL` environment variable.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_1/wide_and_deep.h5
export PRETRAINED_MODEL=$(pwd)/
```

### Run benchmark on Linux
Install the Latest TensorFlow along with model dependencies under requirements.txt.
Set the environment variables and run quickstart script on either Linux or Windows systems. See the list of quickstart scripts for details on the different options.
```
# cd to your AI Reference Models directory
cd models

export TF_USE_LEGACY_KERAS=0
export DATASET_DIR=<path to the Wide & Deep dataset directory>
export PRECISION=fp32
export OUTPUT_DIR=<path to the directory where log files will be written>
export PRETRAINED_MODEL=<pretrained model directory>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Install model specific dependencies:
pip install -r models_v2/tensorflow/wide_deep/inference/cpu/requirements.txt

# Run the quickstart scripts
./models_v2/tensorflow/wide_deep/inference/cpu/<script name>.sh
```

### Run accuracy on Linux
```
# cd to your AI Reference Models directory
cd models

export TF_USE_LEGACY_KERAS=0
export DATASET_DIR=<path to the Wide & Deep dataset directory>
export PRECISION=fp32
export OUTPUT_DIR=<path to the directory where log files will be written>
export PRETRAINED_MODEL=<pretrained model directory>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>
# Install model specific dependencies:
pip install -r models_v2/tensorflow/wide_deep/inference/cpu/requirements.txt

./models_v2/tensorflow/wide_deep/inference/cpu/accuracy.sh
```

### Run benchmark on Windows
Using cmd.exe, run:
```
# cd to your AI Reference Models directory
cd models

set TF_USE_LEGACY_KERAS=0
set PRETRAINED_MODEL=<pretrained model directory>
set DATASET_DIR=<path to the Wide & Deep dataset directory>
set PRECISION=fp32
set OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
set BATCH_SIZE=<customized batch size value>
# Install model specific dependencies:
pip install -r models_v2\tensorflow\wide_deep\inference\cpu\requirements.txt
bash models_v2\tensorflow\wide_deep\inference\cpu\<script name>.sh
```
> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables.
As an example, if the dataset location on Windows is `D:\<user>\widedeep_dataset`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\<user>\widedeep_dataset
> /d/<user>/widedeep_dataset
>```
>Then, set the `DATASET_DIR` environment variable `set DATASET_DIR=/d/<user>/widedeep_dataset`.

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions for the available precisions [FP32](fp32/Advanced.md) [<int8 precision>](<int8 advanced readme link>) [<bfloat16 precision>](<bfloat16 advanced readme link>) for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [Intel® Developer Catalog](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)
  workload container:<br />
  [https://www.intel.com/content/www/us/en/developer/articles/containers/wide-deep-fp32-inference-tensorflow-container.html](https://www.intel.com/content/www/us/en/developer/articles/containers/wide-deep-fp32-inference-tensorflow-container.html).
