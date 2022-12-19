<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* git
* numactl
* wget
* [Protobuf Compilation](https://github.com/tensorflow/models/blob/v1.12.0/research/object_detection/g3doc/installation.md#protobuf-compilation)
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* Cython
* contextlib2
* jupyter
* lxml
* matplotlib
* pillow>=9.3.0
* pycocotools

Clone the [TensorFlow Model Garden](https://github.com/tensorflow/models)
repository using the specified tag, and save the path to the `TF_MODELS_DIR`
environment variable.
```
# Clone the TF models repo
git clone https://github.com/tensorflow/models.git
pushd models
git checkout tags/v1.12.0
export TF_MODELS_DIR=$(pwd)
popd
```

After installing the prerequisites, download and extract the model
package, which includes the pretrained model and scripts needed
to run inference. Set environment variables for the path to
your `DATASET_DIR` (where the coco TF records file is
located) and an `OUTPUT_DIR` where log files will be written, then run a
[quickstart script](#quick-start-scripts).

```
wget <package url>
tar -xzf <package name>
cd <package dir>

export DATASET_DIR=<path to the coco dataset>
export OUTPUT_DIR=<directory where log files will be written>

./quickstart/<script name>.sh
```
