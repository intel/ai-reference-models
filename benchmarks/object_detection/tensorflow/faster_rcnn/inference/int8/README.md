<!--- 0. Title -->
# Faster R-CNN Int8 inference

<!-- 10. Description -->
## Description

This document has instructions for running Faster R-CNN Int8 inference using
Intel Optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

The [COCO validation dataset](http://cocodataset.org) is used in the
Faster R-CNN quickstart scripts. The scripts require that the dataset
has been converted to the TF records format. See the
[COCO dataset](/datasets/coco/README.md) for instructions on downloading
and preprocessing the COCO validation dataset.

Set the `DATASET_DIR` to point to this directory when running Faster R-CNN.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [int8_inference.sh](int8_inference.sh) | Runs batch and online inference using the raw coco dataset |
| [int8_accuracy.sh](int8_accuracy.sh) | Runs inference and evaluates the model's accuracy with coco dataset in `coco_val.record` format |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
* [Kubernetes](#kubernetes)

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
      <p>AI Kit does not currently support TF 1.15.2 models</p>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3.6 or 3.7
        <li>git
        <li>numactl
        <li>wget
        <li><a href="https://github.com/tensorflow/models/blob/v1.12.0/research/object_detection/g3doc/installation.md#protobuf-compilation">Protobuf Compilation</a>
        <li><a href="https://pypi.org/project/intel-tensorflow/1.15.2/">intel-tensorflow==1.15.2</a>
        <li>Cython
        <li>contextlib2
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>pillow>=8.1.2
        <li>pycocotools
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

Running Faster R-CNN Int8 inference also requires cloning the TensorFlow
models repo using the tag specified below. Set the `TF_MODELS_DIR` environment
variable to point to the TensorFlow models directory. Run the
[protobuf-compiler](https://github.com/tensorflow/models/blob/v1.12.0/research/object_detection/g3doc/installation.md#protobuf-compilation)
on the `research` directory.
```
# Clone the TF models repo
git clone https://github.com/tensorflow/models.git tf_models
pushd tf_models
git checkout tags/v1.12.0
export TF_MODELS_DIR=$(pwd)

# Run the protobuf-compiler from the TF models research directory
pushd research
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.
rm protobuf.zip
popd
popd
```

Download the pretrained model and set the `PRETRAINED_MODEL` environment
variable to point to the frozen graph. The quickstart scripts will use this var.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/faster_rcnn_int8_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/faster_rcnn_int8_pretrained_model.pb
```

In addition to the `TF_MODELS_DIR` and `PRETRAINED_MODEL` variables from
above, set environment variables for the path to your `DATASET_DIR` (where
the coco TF records file is located) and an `OUTPUT_DIR` where log files
will be written, then run a [quickstart script](#quick-start-scripts).
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the coco dataset>
export OUTPUT_DIR=<directory where log files will be written>
export TF_MODELS_DIR=<path to the TensorFlow models dir>
export PRETRAINED_MODEL=<path to the pretrained model frozen graph file>

./quickstart/object_detection/tensorflow/faster_rcnn/inference/cpu/int8/<script name>.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/faster-rcnn-int8-inference-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/faster-rcnn-int8-inference-tensorflow-container.html).

