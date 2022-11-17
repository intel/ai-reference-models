<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* <model name> uses the object detection code from the
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


After installing the prerequisites, download and untar the model package.
```
wget <package url>
tar -xzf <package name>
```

Set environment variables for the TensorFlow Model Garden directory to `TF_MODELS_DIR`, the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

To run inference with performance metrics:
```
DATASET_DIR=<path to the coco val2017 raw image directory (ex: /home/user/coco_dataset/val2017)>
OUTPUT_DIR=<directory where log files will be written>
TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

cd <package name>
./quickstart/int8_inference.sh
```

To get accuracy metrics:
```
DATASET_DIR=<path to the COCO validation TF record file (ex: /home/user/coco_output/coco_val.record)>
OUTPUT_DIR=<directory where log files will be written>
TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

cd <package name>
./quickstart/int8_accuracy.sh
```
