<!--- 0. Title -->
# SSD-MobileNet FP32 inference

<!-- 10. Description -->

This document has instructions for running SSD-MobileNet FP32 inference using
Intel-optimized TensorFlow.


<!--- 30. Datasets -->
## Dataset

The [COCO validation dataset](http://cocodataset.org) is used in these
SSD-Mobilenet quickstart scripts. The inference and accuracy quickstart scripts require the dataset to be converted into the TF records format.
See the [COCO dataset](/datasets/coco/README.md) for instructions on
downloading and preprocessing the COCO validation dataset.


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_inference.sh`](fp32_inference.sh) | Runs inference on TF records and outputs performance metrics. |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | Processes the TF records to run inference and check accuracy on the results. |
| [`multi_instance_batch_inference.sh`](multi_instance_batch_inference.sh) | A multi-instance run that uses all the cores for each socket for each instance with a batch size of 448 and synthetic data. |
| [`multi_instance_online_inference.sh`](multi_instance_online_inference.sh) | A multi-instance run that uses 4 cores per instance with a batch size of 1. Uses synthetic data if no `DATASET_DIR` is set. |

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

For more information on the dependencies, see the documentation on [prerequisites](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
in the TensorFlow models repo.

Download the pretrained model and set the `PRETRAINED_MODEL` environment
variable to the path of the frozen graph.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb
export PRETRAINED_MODEL=$(pwd)/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb
```

To run inference, navigate to your Model Zoo directory, set environment variables
with the path to the dataset and an output directory, and then run a
[quickstart script](#quick-start-scripts).
```
# cd to your model zoo directory
cd models

export PRETRAINED_MODEL=<path to the downloaded frozen graph>
export DATASET_DIR=<path to the coco tf record file>
export OUTPUT_DIR=<directory where log files will be written>

./quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/fp32/<script name>.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/ssd-mobilenet-fp32-inference-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/ssd-mobilenet-fp32-inference-tensorflow-container.html).

