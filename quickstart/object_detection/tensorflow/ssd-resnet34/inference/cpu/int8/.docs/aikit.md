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
        <li>build-essential
        <li>git
        <li>libgl1-mesa-glx
        <li>libglib2.0-0
        <li>numactl
        <li>python3-dev
        <li>wget
        <li>Cython
        <li>contextlib2
        <li>horovod==0.19.1
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>numpy>=1.17.4
        <li>opencv-python
        <li>pillow>=8.1.2
        <li>pycocotools
        <li>tensorflow-addons==0.11.0
        <li>Activate the tensorflow 2.5.0 conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit on Linux you will need:</p>
      <ul>
        <li>Python 3
        <li>build-essential
        <li>git
        <li>libgl1-mesa-glx
        <li>libglib2.0-0
        <li>numactl
        <li>python3-dev
        <li>wget
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>Cython
        <li>contextlib2
        <li>horovod==0.19.1
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>numpy>=1.17.4
        <li>opencv-python
        <li>pillow>=8.1.2
        <li>pycocotools
        <li>tensorflow-addons==0.11.0
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit on Windows you will need:</p>
      <ul>
        <li><a href="/docs/general/tensorflow/Windows.md">Intel Model Zoo on Windows Systems prerequisites</a>
        <li>build-essential
        <li>libgl1-mesa-glx
        <li>libglib2.0-0
        <li>python3-dev
        <li>Cython
        <li>contextlib2
        <li>horovod==0.19.1
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>numpy>=1.17.4
        <li>opencv-python
        <li>pillow>=8.1.2
        <li>pycocotools
        <li>tensorflow-addons==0.11.0
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

In addition to the libraries above, SSD-ResNet34 uses the
[TensorFlow models](https://github.com/tensorflow/models) and
[TensorFlow benchmarks](https://github.com/tensorflow/benchmarks)
repositories. Clone the repositories using the commit ids specified
below and set the `TF_MODELS_DIR` to point to the folder where the models
repository was cloned:
```
# Clone the TensorFlow models repo
git clone https://github.com/tensorflow/models.git tf_models
cd tf_models
git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
export TF_MODELS_DIR=$(pwd)
cd ..

# Clone the TensorFlow benchmarks repo
git clone --single-branch https://github.com/tensorflow/benchmarks.git ssd-resnet-benchmarks
cd ssd-resnet-benchmarks
git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
cd ..
```

Download the SSD-ResNet34 pretrained model for either the 300x300 or 1200x1200
input size, depending on which [quickstart script](#quick-start-scripts) you are
going to run. Set the `PRETRAINED_MODEL` environment variable for the path to the
pretrained model that you'll be using.
If you run on Windows, please use a browser to download the pretrained model using the link below.
For Linux, run:
```
# ssd-resnet34 300x300
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_int8_bs1_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/ssd_resnet34_int8_bs1_pretrained_model.pb

# ssd-resnet34 1200x1200
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_int8_1200x1200_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/ssd_resnet34_int8_1200x1200_pretrained_model.pb
```

After installing the prerequisites and cloning the models and benchmarks
repos, and downloading the pretrained model, set the required environment
variables. Set an environment variable for the path to an `OUTPUT_DIR`
where log files will be written. If the accuracy test is being run, then
also set the `DATASET_DIR` to point to the folder where the COCO dataset
`validation-00000-of-00001` file is located.  Once the environment variables
are set, you can then run a [quickstart script](#quick-start-scripts) from the
Model Zoo on either Linux or Windows.

### Run on Linux
```
# cd to your model zoo directory
cd models

# set environment variables
export DATASET_DIR=<directory with the validation-*-of-* files (for accuracy testing only)>
export TF_MODELS_DIR=<path to the TensorFlow Models repo>
export PRETRAINED_MODEL=<path to the 300x300 or 1200x1200 pretrained model pb file>
export OUTPUT_DIR=<directory where log files will be written>

./quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/int8/<script name>.sh
```

### Run on Windows
Using `cmd.exe`, run:
```
# cd to your model zoo directory
cd models

set PRETRAINED_MODEL=<path to the 300x300 or 1200x1200 pretrained model pb file>
set DATASET_DIR=<directory with the validation-*-of-* files (for accuracy testing only)>
set OUTPUT_DIR=<directory where log files will be written>
set TF_MODELS_DIR=<path to the TensorFlow Models repo>

bash quickstart\object_detection\tensorflow\ssd-resnet34\inference\cpu\int8\<script name>.sh
```
> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables. 
As an example, if the dataset location on Windows is `D:\user\coco_dataset`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\coco_dataset
> /d/user/coco_dataset
>```
>Then, set the `DATASET_DIR` environment variable `set DATASET_DIR=/d/user/coco_dataset`.
