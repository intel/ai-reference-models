<!--- 0. Title -->
# SSD-MobileNet inference

<!-- 10. Description -->
## Description

This document has instructions for running SSD-MobileNet inference using
Intel-optimized TensorFlow.
<!--- 30. Datasets -->
## Datasets

The [COCO validation dataset](http://cocodataset.org) is used in these
SSD-Mobilenet quickstart scripts. The accuracy quickstart script require the dataset to be converted into the TF records format.
See the [COCO dataset](https://github.com/IntelAI/models/tree/master/datasets/coco) for instructions on
downloading and preprocessing the COCO validation dataset.

Set the `DATASET_DIR` to point to the dataset directory that contains the TF records file `coco_val.record` when running SSD-MobileNet accuracy script.
<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`inference.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference.sh) | Runs inference and outputs performance metrics. Uses synthetic data if no `DATASET_DIR` is set. Supported versions are (fp32, int8, bfloat16) |
| [`accuracy.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/accuracy.sh) | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required). Supported versions are (fp32, int8, bfloat16, bfloat32). |
| [`inference_throughput_multi_instance.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference_throughput_multi_instance.sh) | A multi-instance run that uses all the cores for each socket for each instance with a batch size of 448 and synthetic data. Supported versions are (fp32, int8, bfloat16, bfloat32) |
| [`inference_realtime_multi_instance.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference_realtime_multi_instance.sh) | A multi-instance run that uses 4 cores per instance with a batch size of 1. Uses synthetic data if no `DATASET_DIR` is set. Supported versions are (fp32, int8, bfloat16, bfloat32) |

<!--- 50. AI Tools -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Tools](/docs/general/tensorflow/AITools.md):

<table>
  <tr>
    <th>Setup using AI Tools on Linux</th>
    <th>Setup without AI Tools on Linux</th>
    <th>Setup without AI Tools on Windows</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Tools on Linux you will need:</p>
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
        <li>intel-extension-for-tensorflow (only required when using onednn graph optimization)
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Tools on Linux you will need:</p>
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
        <li>intel-extension-for-tensorflow (only required when using onednn graph optimization)
        <li>A clone of the AI Reference Models repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Tools on Windows you will need:</p>
      <ul>
        <li><a href="/docs/general/Windows.md">Intel AI Reference Models on Windows Systems prerequisites</a>
        <li>build-essential
        <li>Cython
        <li>contextlib2
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>pillow>=9.3.0
        <li>pycocotools
        <li>A clone of the AI Reference Models repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

For more information on the dependencies, see the documentation on [prerequisites](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
in the TensorFlow models repo.

Download the pretrained model and set the `PRETRAINED_MODEL` environment
variable to the path of the frozen graph. If you run on Windows, please use a browser to download the pretrained model using the link below.
For Linux, run:
```
# FP32, BFloat16 and BFloat32 Pretrained model
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb
export PRETRAINED_MODEL=$(pwd)/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb

# Int8 Pretrained model
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssdmobilenet_int8_pretrained_model_combinedNMS_s8.pb
export PRETRAINED_MODEL=$(pwd)/ssdmobilenet_int8_pretrained_model_combinedNMS_s8.pb

# Int8 Subgraph model (Int8 Accuracy only)
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_1/ssdmobilenet_preprocess.pb
export INPUT_SUBGRAPH=$(pwd)/ssdmobilenet_preprocess.pb

# Int8 Pretrained model for OneDNN Graph (Only used when the plugin Intel Extension for Tensorflow is installed, as OneDNN Graph optimization is enabled by default at this point)
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_12_0/ssd_mb_itex_int8.pb
export PRETRAINED_MODEL=$(pwd)/ssd_mb_itex_int8.pb
```

After installing the prerequisites and downloading the pretrained model, set the environment variables and for the `DATASET_DIR` use COCO raw dataset directory or tf_records file based on whether you run inference or accuracy scripts.
Navigate to your AI Reference Models directory and then run a [quickstart script](#quick-start-scripts) on either Linux or Windows.

### Run on Linux
```
# cd to your AI Reference Models directory
cd models

export PRETRAINED_MODEL=<path to the downloaded frozen graph>
export DATASET_DIR=<path to the coco tf record file>
export PRECISION=<set the precision to "int8" or "fp32" or "bfloat16" or "bfloat32" >
export OUTPUT_DIR=<path to the directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/<script name>.sh
```

### Run on Windows
Using `cmd.exe`, run:
```
# cd to your AI Reference Models directory
cd models

set PRETRAINED_MODEL=<path to the pretrained model pb file>
set DATASET_DIR=<path to the coco tf record file>
set OUTPUT_DIR=<directory where log files will be written>
set PRECISION=<set the precision to "int8" or "fp32" or "bfloat16">
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
set BATCH_SIZE=<customized batch size value>

# Run a quickstart script (inference.sh)
bash quickstart\object_detection\tensorflow\ssd-mobilenet\inference\cpu\inference.sh
```
> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables. 
As an example, if the dataset location on Windows is `D:\user\coco_dataset\coco_val.record`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\coco_dataset\coco_val.record
> /d/user/coco_dataset/coco_val.record
>```
>Then, set the `DATASET_DIR` environment variable `set DATASET_DIR=/d/user/coco_dataset/coco_val.record`.

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions for the available precisions [FP32](fp32/Advanced.md) [Int8](int8/Advanced.md) [BFloat16](bfloat16/Advanced.md) for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [Intel® Developer Catalog](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)
  workload container:<br />
  [https://www.intel.com/content/www/us/en/developer/articles/containers/ssd-mobilenet-fp32-inference-tensorflow-container.html](https://www.intel.com/content/www/us/en/developer/articles/containers/ssd-mobilenet-fp32-inference-tensorflow-container.html).

