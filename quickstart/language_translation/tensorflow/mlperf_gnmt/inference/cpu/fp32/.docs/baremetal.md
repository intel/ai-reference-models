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
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

wget <package url>
tar -xzf <package name>
cd <package dir>

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
