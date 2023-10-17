<!--- 0. Title -->
# DenseNet 169 inference

<!-- 10. Description -->
## Description

This document has instructions for running DenseNet 169 inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to this directory when running DenseNet 169.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`online_inference.sh`](/quickstart/image_recognition/tensorflow/densenet169/inference/cpu/online_inference.sh) | Runs online inference (batch_size=1). |
| [`batch_inference.sh`](/quickstart/image_recognition/tensorflow/densenet169/inference/cpu/batch_inference.sh) | Runs batch inference (batch_size=100). |
| [`accuracy.sh`](/quickstart/image_recognition/tensorflow/densenet169/inference/cpu/accuracy.sh) | Measures the model accuracy (batch_size=100). |

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
        <li>Activate the tensorflow conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Tools on Linux you will need:</p>
      <ul>
        <li>Python 3
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>git
        <li>numactl
        <li>wget
        <li>A clone of the AI Reference Models repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Tools on Windows you will need:</p>
      <ul>
        <li><a href="/docs/general/Windows.md">Intel AI Reference Models on Windows Systems prerequisites</a>
        <li>A clone of the AI Reference Models repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After your setup is complete, download the pretrained model. The path to this file
should be set to the `PRETRAINED_MODEL` environment variable before running the quickstart scripts. 
If you run on Windows, please use a browser to download the pretrained model using the link below.
For Linux, run:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/densenet169_fp32_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/densenet169_fp32_pretrained_model.pb
```
Set the environment variables and run quickstart script on either Linux or Windows systems. See the [list of quickstart scripts](#quick-start-scripts) for details on the different options.

### Run on Linux
```
# cd to your AI Reference Models directory
cd models

export DATASET_DIR=<path to the dataset>
export PRECISION=fp32
export OUTPUT_DIR=<path to the directory where log files will be written>
export PRETRAINED_MODEL=<path to the frozen graph that you downloaded>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run a script for your desired usage
./quickstart/image_recognition/tensorflow/densenet169/inference/cpu/<script name>.sh
```

### Run on Windows
Using a `cmd.exe` run:
```
# cd to your AI Reference Models directory
cd models

set PRETRAINED_MODEL=<path to the frozen graph downloaded above>
set DATASET_DIR=<path to the ImageNet TF records>
set PRECISION=fp32
set OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
set BATCH_SIZE=<customized batch size value>

bash quickstart\image_recognition\tensorflow\densenet169\inference\cpu\<script name>.sh
```
> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables. 
As an example, if the dataset location on Windows is `D:\user\ImageNet`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\ImageNet
> /d/user/ImageNet
>```
>Then, set the `DATASET_DIR` environment variable `set DATASET_DIR=/d/user/ImageNet`.

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions for the available precisions [FP32](fp32/Advanced.md) [<int8 precision>](<int8 advanced readme link>) [<bfloat16 precision>](<bfloat16 advanced readme link>) for calling the `launch_benchmark.py` script directly.
