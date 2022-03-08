<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit on Linux</th>
    <th>Setup without AI Kit on Linux</th>
    <th>Setup without AI Kit on Windows</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Kit on Linux you will need:</p>
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
      <p>To run without AI Kit on Linux you will need:</p>
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
    <td>
      <p>To run without AI Kit on Windows you will need:</p>
      <ul>
        <li><a href="/docs/general/tensorflow/Windows.md">Intel Model Zoo on Windows Systems prerequisites</a>
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
environment variable to point to the frozen graph file.
If you run on Windows, please use a browser to download the pretrained model using the link below.
For Linux, run:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/rfcn_resnet101_int8_coco_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/rfcn_resnet101_int8_coco_pretrained_model.pb
```

RFCN uses the object detection code from the [TensorFlow Model Garden](https://github.com/tensorflow/models).
Clone this repo with the SHA specified below and apply the patch from the model zoo directory.
Set the `TF_MODELS_DIR` environment variable to the path of your clone of the TF Model Garden.
```
# Clone the TensorFlow Model Garden
git clone https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b

# Apply the TF2 patch from the model zoo repo directory
git apply --ignore-space-change --ignore-whitespace <model zoo directory>/models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch

# Set the TF_MODELS_DIR env var
export TF_MODELS_DIR=$(pwd)
```

### Run on Linux
Download and install [Google Protobuf version 3.3.0](https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip), and
run [protobuf compilation](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#protobuf-compilation).
```
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

### Run on Windows
* Download and install [Google Protobuf version 3.4.0]((https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0)) for Windows in addition to the above listed dependencies.
Download and extract [protoc-3.4.0-win32.zip](https://github.com/protocolbuffers/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip)
Navigate to the `research` directory in `TF_MODELS_DIR` and install Google Protobuf:
```
set TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>
cd %TF_MODELS_DIR%\research
“C:\<user>\protoc-3.4.0-win32\bin\protoc.exe” object_detection/protos/*.proto --python_out=.
```

After installing the prerequisites and cloning the TensorFlow models repo, and downloading the pretrained model,
set the environment variables for the paths to your `PRETRAINED_MODEL`, an `OUTPUT_DIR` where log files will be written,
TF_MODELS_DIR, and `DATASET_DIR` for COCO raw dataset directory or tf_records file based on whether you run inference or accuracy scripts.
Navigate to your model zoo directory and then run a [quickstart script](#quick-start-scripts).
```
# cd to your model zoo directory
cd models

set PRETRAINED_MODEL=<path to the frozen graph downloaded above>
set DATASET_DIR=<path to COCO raw dataset directory or tf_records file based on whether you run inference or accuracy scripts>
set OUTPUT_DIR=<directory where log files will be written>
set TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>

bash quickstart\object_detection\tensorflow\rfcn\inference\cpu\int8\<script name>.sh
```
> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables. 
As an example, if the dataset location on Windows is `D:\user\coco_dataset\val2017`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\coco_dataset\val2017
> /d/user/coco_dataset/val2017
>```
>Then, set the `DATASET_DIR` environment variable `set DATASET_DIR=/d/user/coco_dataset/val2017`.
