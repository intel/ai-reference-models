# RFCN

The following examples are available for RFCN using a model package:
* [FP32 Inference](#fp32-inference)

Note that the [COCO dataset](http://cocodataset.org) is used in the RFCN examples. The inference
examples use raw images, and the accuracy examples require the dataset to be converted into the
TF records format. See the document <HERE> for instructions on downloading and preprocessing the 
COCO dataset.

## FP32 Inference

### Examples

* fp32_inferece: Runs inference on a directory of raw images for 500 steps and outputs performance metrics
* fp32_accuracy: Processes the TF records to run inference and check accuracy on the results.

These examples can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
* [Kubernetes](#kubernetes)

#### Bare Metal

To run on bare metal, prerequisites to run the model zoo scripts must be installed on in your environment <LINK>.

Download and untar the RFCN FP32 inference model package:

```
wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/rfcn-fp32-inference.tar.gz
tar -xvf rfcn_fp32_inference.tar.gz
```

In addition to the general model zoo requirements, RFCN uses the object detection code from the
[TensorFlow Model Garden](https://github.com/tensorflow/models). Clone this repo with the SHA specified
below and apply the patch from the RFCN FP32 inference model package to run with TF2.

```
git clone https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b
git apply ../rfcn_fp32_inference/models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch
```

You must also install the dependencies and run the protobuf compilation described in the
[object detection installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#installation)
from the [TensorFlow Model Garden](https://github.com/tensorflow/models) repository.

Once your environment is setup, navigate back to the directory that contains the RFCN FP32 inference
model package, set environment variables pointing to your dataset and output directories, and then run
an example.


To run inference with performance metrics:

```
DATASET_DIR=<path to the raw coco images>
OUTPUT_DIR=<directory where log files will be written>

examples/fp32_inference
```

To get accuracy metrics:
```
DATASET_DIR=<directory where your TF records file is located>
TF_RECORD_FILE=<name of your TF records file>
OUTPUT_DIR=<directory where log files will be written>

examples/fp32_accuracy
```


#### Docker

When running in docker, the `tf-rfcn-fp32-inference` container includes the libraries and the model
package, which are needed to run RFCN FP32 inference. To run the examples, you'll need to provide
volume mounts for the COCO dataset and an output directory where log files will be written.

To run inference with performance metrics:

```
DATASET_DIR=<path to the raw coco images>
OUTPUT_DIR=<directory where log files will be written>

docker run \
        --env DATASET_DIR=${DATASET_DIR} \
        --env OUTPUT_DIR=${OUTPUT_DIR} \
        --env http_proxy=${http_proxy} --env https_proxy=${https_proxy} \
        --volume ${DATASET_DIR}:${DATASET_DIR} \
        --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
        --privileged --init -it \
        tf-rfcn-fp32-inference:2.1.0 \
        /bin/bash examples/fp32_inference
```

To get accuracy metrics:

```
DATASET_DIR=<directory where your TF records file is located>
TF_RECORD_FILE=<name of your TF records file>
OUTPUT_DIR=<directory where log files will be written>

docker run \
        --env DATASET_DIR=${DATASET_DIR} \
        --env OUTPUT_DIR=${OUTPUT_DIR} \
        --env TF_RECORD_FILE=${TF_RECORD_FILE} \
        --env http_proxy=${http_proxy} --env https_proxy=${https_proxy} \
        --volume ${DATASET_DIR}:${DATASET_DIR} \
        --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
        --privileged --init -it \
        tf-rfcn-fp32-inference:2.1.0 \
        /bin/bash examples/fp32_accuracy
```
