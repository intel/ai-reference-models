<!--- 0. Title -->
# MLPerf GNMT inference

<!-- 10. Description -->
## Description

This document has instructions for running MLPerf GNMT inference using
Intel-optimized TensorFlow.

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
| [`online_inference.sh`](/quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/online_inference.sh) | Runs online inference (batch_size=1). |
| [`batch_inference.sh`](/quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/batch_inference.sh) | Runs batch inference (batch_size=32). |
| [`accuracy.sh`](/quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/accuracy.sh) | Runs accuracy |

<!--- 50. AI Tools -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Tools](/docs/general/tensorflow/AITools.md):

<table>
  <tr>
    <th>Setup using AI Tools</th>
    <th>Setup without AI Tools</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Tools you will need:</p>
      <ul>
        <li>git
        <li>numactl
        <li>pip
        <li>wget
        <li><a href="https://bazel.build/">Bazel</a> to build tensorflow addons
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Tools you will need:</p>
      <ul>
        <li>Python 3
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>git
        <li>numactl
        <li>pip
        <li>wget
        <li><a href="https://bazel.build/">Bazel</a> to build tensorflow addons
        <li>A clone of the AI Reference Models repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After installing the prerequisites, download the pretrained model and set
the `PRETRAINED_MODEL` environment variable to the .pb file path:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/mlperf_gnmt_fp32_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/mlperf_gnmt_fp32_pretrained_model.pb
```

MLPerf GNMT requires TensorFlow addons to be built with a patch from the model
zoo. The snippet below shows how to clone the addons repo, apply the patch, and
then build and install the TensorFlow addons wheel.
```
# TensorFlow addons (r0.5) build and installation instructions:
#   Clone TensorFlow addons (r0.5) and apply a patch: A patch file
#   is attached in Intel AI Reference Models MLpref GNMT model scripts,
#   it fixes TensorFlow addons (r0.5) to work with TensorFlow
#   version 2.11, and prevents TensorFlow 2.0.0 to be installed
#   by default as a required dependency.
git clone --single-branch --branch=r0.5 https://github.com/tensorflow/addons.git
cd addons
git apply ../models/language_translation/tensorflow/mlperf_gnmt/gnmt-fix.patch

#   Build TensorFlow addons source code and create TensorFlow addons
#   pip wheel. Use bazel 6.0.0 version :
#   Answer yes to questions while running configure.sh
bash configure.sh
bazel build --enable_runfiles build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip install artifacts/tensorflow_addons-*.whl --no-deps
```

Once that has been completed, ensure you have the required environment variables
set, and then run a quickstart script.

```
# cd to your AI Reference Models directory
cd models

# Set env var paths
export DATASET_DIR=<path to the dataset>
export PRECISION=fp32
export OUTPUT_DIR=<path to the directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model frozen graph>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run a quickstart script
./quickstart/language_translation/tensorflow/mlperf_gnmt/inference/cpu/<script name>.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions for the available precisions [FP32](fp32/Advanced.md) [<int8 precision>](<int8 advanced readme link>) [<bfloat16 precision>](<bfloat16 advanced readme link>) for calling the `launch_benchmark.py` script directly.
