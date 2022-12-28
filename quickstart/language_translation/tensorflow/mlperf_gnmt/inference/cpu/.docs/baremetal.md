<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit</th>
    <th>Setup without AI Kit</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Kit you will need:</p>
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
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>git
        <li>numactl
        <li>pip
        <li>wget
        <li><a href="https://bazel.build/">Bazel</a> to build tensorflow addons
        <li>A clone of the Model Zoo repo<br />
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

<model name> requires TensorFlow addons to be built with a patch from the model
zoo. The snippet below shows how to clone the addons repo, apply the patch, and
then build and install the TensorFlow addons wheel.
```
# TensorFlow addons (r0.5) build and installation instructions:
#   Clone TensorFlow addons (r0.5) and apply a patch: A patch file
#   is attached in Intel Model Zoo MLpref GNMT model scripts,
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
# cd to your model zoo directory
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
