<!--- 0. Title -->
# EDSRx3 inference

<!-- 10. Description -->
## Description

This document has instructions for running Enhanced Deep Super-Resolution Network inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets
EDSRx3 is a super resolution model trained on [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/). In Model Zoo, to perform accuracy or benchmarking using DIV2K dataset, the code takes care of it by installing the corresponding Tensorflow-Dataset and will be used for future runs.

During Benchmarking, if the use of DIV2K dataset should be done, set env variable USE_REAL_DATA=True and DIV2K dataset validation images will be used to perform benchmarking.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs realtime inference using a default `batch_size=32` for the specified precision (fp32). To run inference for throughtput, set `BATCH_SIZE` environment variable. |
| `accuracy.sh` | Measures the model accuracy (batch_size=32) for the specified precision (fp32). |


## Run the model
Install the Intel-optimized TensorFlow along with model dependencies under [requirements.txt](../../../../../models/super_resolution/tensorflow/edsr/inference/requirements.txt).

After finishing the setup above, download the pretrained model based on `PRECISION`and set the
`PRETRAINED_MODEL` environment var to the path to the frozen graph.
```
# FP32 Pretrained model:
git clone https://github.com/Saafke/EDSR_Tensorflow.git
cd EDSR_Tensorflow/models
export PRETRAINED_MODEL=$(pwd)/EDSR_x3.pb
```

Set the environment variables and run quickstart script on Linux systems. See the [list of quickstart scripts](#quick-start-scripts) for details on the different options.

### Run on Linux:
```
# cd to your model zoo directory
cd models

export PRETRAINED_MODEL=<path to the frozen graph downloaded above>
export PRECISION=<set the precision to "fp32">
export OUTPUT_DIR=<path to the directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>
# To run benchmarking with DIV2K dataset val images, set USE_REAL_DATA env variable.
export USE_REAL_DATA=False

./quickstart/super_resolution/tensorflow/edsr/inference/cpu/<script name>.sh
```
