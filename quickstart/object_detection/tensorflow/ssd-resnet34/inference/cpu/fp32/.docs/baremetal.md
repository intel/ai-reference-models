<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* build-essential
* git
* libgl1-mesa-glx
* libglib2.0-0
* numactl
* python3-dev
* wget
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* Cython
* contextlib2
* horovod==0.19.1
* jupyter
* lxml
* matplotlib
* numpy>=1.17.4
* opencv-python
* pillow>=9.3.0
* pycocotools
* tensorflow-addons==0.11.0

The [TensorFlow models](https://github.com/tensorflow/models) and
[benchmarks](https://github.com/tensorflow/benchmarks) repos are used by
<model name> <precision> <mode>. Clone those at the git SHAs specified
below and set the `TF_MODELS_DIR` environment variable to point to the
directory where the models repo was cloned.

```
git clone --single-branch https://github.com/tensorflow/models.git tf_models
git clone --single-branch https://github.com/tensorflow/benchmarks.git ssd-resnet-benchmarks
cd tf_models
export TF_MODELS_DIR=$(pwd)
git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
cd ../ssd-resnet-benchmarks
git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
cd ..
```

After installing the prerequisites and cloning the models and benchmarks
repos, download and untar the model package.
Set environment variables for the path to your `DATASET_DIR` (for accuracy
testing only -- inference benchmarking uses synthetic data) and an
`OUTPUT_DIR` where log files will be written, then run a
[quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the dataset (for accuracy testing only)>
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

wget <package url>
tar -xzf <package name>
cd <package dir>

./quickstart/<script name>.sh
```
