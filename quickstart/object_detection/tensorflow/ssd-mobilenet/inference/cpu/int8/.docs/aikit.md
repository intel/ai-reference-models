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
        <li>numactl
        <li>wget
        <li>build-essential
        <li>Cython
        <li>contextlib2
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>pillow>=8.1.2
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
        <li>pycocotools
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

For more information see the documentation on [prerequisites](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
in the TensorFlow models repo.

Download the pretrained model and set the `PRETRAINED_MODEL` environment
variable to the path of the frozen graph.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssdmobilenet_int8_pretrained_model_combinedNMS_s8.pb
export PRETRAINED_MODEL=$(pwd)/ssdmobilenet_int8_pretrained_model_combinedNMS_s8.pb
```

To run <mode>, navigate to your Model Zoo directory, set environment variables
with the path to the dataset and an output directory, and then run a
[quickstart script](#quick-start-scripts).
```
# cd to your model zoo directory
cd models

export PRETRAINED_MODEL=<path to the pretrained model pb file>
export DATASET_DIR=<path to the coco tf record file>
export OUTPUT_DIR=<directory where log files will be written>

./quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/int8/<script name>.sh
```
