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
        <li>wget
        <li>build-essential
        <li>Cython
        <li>contextlib2
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>pillow>=8.1.2
        <li>protobuf-compiler
        <li>pycocotools
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li>git
        <li>numactl
        <li>wget
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>build-essential
        <li>Cython
        <li>contextlib2
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>pillow>=8.1.2
        <li>protobuf-compiler
        <li>pycocotools
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

For more information on the required dependencies, see the documentation on [prerequisites](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
in the TensorFlow models repo.

Download the pretrained model and set the `PRETRAINED_MODEL`
environment variable to point to the frozen graph file:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/rfcn_resnet101_int8_coco_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/rfcn_resnet101_int8_coco_pretrained_model.pb
```

<model name> uses the object detection code from the [TensorFlow Model Garden](https://github.com/tensorflow/models).
Clone this repo with the SHA specified below and apply the patch from the model zoo directory.
Set the `TF_MODELS_DIR` environment variable to the path of your clone of the TF Model Garden and
run [protobuf compilation](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#protobuf-compilation).
```
# Clone the TensorFlow Model Garden
git clone https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b

# Apply the TF2 patch from the model zoo repo directory
git apply <model zoo directory>/models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch

# Set the TF_MODELS_DIR env var
export TF_MODELS_DIR=$(pwd)

# Protobuf compilation from the TF models research directory
cd research
protoc object_detection/protos/*.proto --python_out=.
cd ../..
```

Once your environment is setup, navigate back to your Model Zoo directory. Ensure that
you have set environment variables pointing to the TensorFlow Model Garden repo, the dataset,
and output directories, and then run a quickstart script.

To run inference with performance metrics:
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the coco val2017 raw image directory (ex: /home/user/coco_dataset/val2017)>
export OUTPUT_DIR=<directory where log files will be written>
export TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>

./quickstart/object_detection/tensorflow/rfcn/inference/cpu/int8/int8_inference.sh
```

To get accuracy metrics:
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to TF record file (ex: /home/user/coco_output/coco_val.record)>
export OUTPUT_DIR=<directory where log files will be written>
export TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>

./quickstart/object_detection/tensorflow/rfcn/inference/cpu/int8/int8_accuracy.sh
```
