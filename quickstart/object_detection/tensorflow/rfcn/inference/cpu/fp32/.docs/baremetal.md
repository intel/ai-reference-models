<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* build-essential
* Cython
* contextlib2
* jupyter
* lxml
* matplotlib
* numpy>=1.17.4
* pillow>=8.1.2
* pycocotools

For more information, see the documentation on [prerequisites](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
in the TensorFlow models repo.

Download and untar the <model name> <precision> inference model package:

```
wget <package url>
tar -xvf <package name>
```

In addition to the general model zoo requirements, <model name> uses the object detection code from the
[TensorFlow Model Garden](https://github.com/tensorflow/models). Clone this repo with the SHA specified
below and apply the patch from the <model name> <precision> <mode> model package to run with TF2.

```
git clone https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b
git apply ../<package dir>/models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch
```

You must also install the dependencies and run the protobuf compilation described in the
[object detection installation instructions](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
from the [TensorFlow Model Garden](https://github.com/tensorflow/models) repository.

Once your environment is setup, navigate back to the directory that contains the <model name> <precision> <mode>
model package, set environment variables pointing to your dataset and output directories, and then run
a quickstart script.

To run inference with performance metrics:
```
DATASET_DIR=<path to the coco val2017 raw image directory (ex: /home/user/coco_dataset/val2017)>
OUTPUT_DIR=<directory where log files will be written>
TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/fp32_inference.sh
```

To get accuracy metrics:
```
DATASET_DIR=<path to TF record file (ex: /home/user/coco_output/coco_val.record)>
OUTPUT_DIR=<directory where log files will be written>
TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/fp32_accuracy.sh
```
