<!--- 0. Title -->
# MLPerf GNMT FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running MLPerf GNMT FP32 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[mlperf-gnmt-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/mlperf-gnmt-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

Download and unzip the MLPerf GNMT model benchmarking data.

```
wget https://zenodo.org/record/2531868/files/gnmt_inference_data.zip
unzip gnmt_inference_data.zip
export DATASET_DIR=$(pwd)/nmt/data
```

Set the `DATASET_DIR` to point as instructed above  when running MLPerf GNMT.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](/quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/fp32/fp32_online_inference.sh) | Runs online inference (batch_size=1). |
| [`fp32_batch_inference.sh`](/quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/fp32/fp32_batch_inference.sh) | Runs batch inference (batch_size=32). |
| [`fp32_accuracy.sh`](/quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/fp32/fp32_accuracy.sh) | Runs accuracy |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* tensorflow-addons - Instructions are provided later
   

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/mlperf-gnmt-fp32-inference.tar.gz
tar -xzf mlperf-gnmt-fp32-inference.tar.gz
cd mlperf-gnmt-fp32-inference

# TensorFlow addons (r0.5) build and installation instructions:
#   Clone TensorFlow addons (r0.5) and apply a patch: A patch file
#   is attached in Intel Model Zoo MLpref GNMT model scripts,
#   it fixes TensorFlow addons (r0.5) to work with TensorFlow 
#   version 2.3, and prevents TensorFlow 2.0.0 to be installed 
#   by default as a required dependency.

git clone --single-branch --branch=r0.5 https://github.com/tensorflow/addons.git
cd addons
git apply ../models/language_translation/tensorflow/mlperf_gnmt/gnmt-v0.5.2.patch

#   Build TensorFlow addons source code and create TensorFlow addons
#   pip wheel. Use bazel 3.0.0 version :

bash configure.sh  # answer yes to questions while running this script
bazel build --enable_runfiles build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip install artifacts/tensorflow_addons-*.whl --no-deps

cd .. # back to package dir
./quickstart/<script name>.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
MLPerf GNMT FP32 inference. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory.

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/language-translation:tf-latest-mlperf-gnmt-fp32-inference \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

