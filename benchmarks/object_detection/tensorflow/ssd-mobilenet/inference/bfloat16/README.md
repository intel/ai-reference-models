<!--- 0. Title -->
# SSD-MobileNet BFloat16 inference

<!-- 10. Description -->

This document has instructions for running SSD-MobileNet BFloat16 inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

The [COCO validation dataset](http://cocodataset.org) is used in these
SSD-Mobilenet quickstart scripts. The accuracy quickstart script require the dataset to be converted into the TF records format.
See the [COCO dataset](https://github.com/IntelAI/models/tree/master/datasets/coco) for instructions on
downloading and preprocessing the COCO validation dataset.

Set the `DATASET_DIR` to point to the dataset directory that contains the TF records file `coco_val.record` when running <model name> accuracy script.


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32 or bfloat16). Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32 or bfloat16). Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32 or bfloat16). |

<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit on Linux</th>
    <th>Setup without AI Kit on Linux</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Kit on Linux you will need:</p>
      <ul>
        <li>numactl
        <li>wget
        <li>build-essential
        <li>Cython
        <li>contextlib2
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>pillow>=9.3.0
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
        <li>pillow>=9.3.0
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
For Linux, run:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb
export PRETRAINED_MODEL=$(pwd)/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb
```

After installing the prerequisites and downloading the pretrained model, set the environment variables for the paths to your `PRETRAINED_MODEL`, an `OUTPUT_DIR` where log files will be written, `DATASET_DIR` for COCO raw dataset directory or tf_records file based on whether you run inference or accuracy scripts and `PRECISION` to bfloat16.
Navigate to your model zoo directory and then run a [quickstart script](#quick-start-scripts) on Linux.

### Run on Linux
```
# cd to your model zoo directory
cd models

# Set environment variables
export PRETRAINED_MODEL=<path to the downloaded frozen graph>
export DATASET_DIR=<path to the directory where coco tf record file is stored>
export OUTPUT_DIR=<path to the directory where log files will be written>
export PRECISION=bfloat16
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run a quickstart script (for example, inference realtime.sh)
./quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/<script name>.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
