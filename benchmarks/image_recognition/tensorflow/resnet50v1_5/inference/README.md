<!--- 0. Title -->
# ResNet50 v1.5 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 v1.5 inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs realtime inference using a default `batch_size=1` for the specified precision (int8, fp32, bfloat16 or fp16). To run inference for throughtput, set `BATCH_SIZE` environment variable. |
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, int8, bfloat16, fp16, bfloat32) with 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_realtime_weightsharing.sh` | Runs multi instance realtime inference with weight sharing for the specified precision (int8 or bfloat16) with 1500 steps and 100 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, int8, bfloat16, fp16, bfloat32) with 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8, bfloat16, fp16, bfloat32). |

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
        <li>numactl
        <li>wget
        <li>openmpi-bin (only required for multi-instance)
        <li>openmpi-common (only required for multi-instance)
        <li>openssh-client (only required for multi-instance)
        <li>openssh-server (only required for multi-instance)
        <li>libopenmpi-dev (only required for multi-instance)
        <li>horovod==0.27.0 (only required for multi-instance)
        <li>Activate the tensorflow conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit on Linux you will need:</p>
      <ul>
        <li>Python 3
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>git
        <li>numactl
        <li>wget
        <li>openmpi-bin (only required for multi-instance)
        <li>openmpi-common (only required for multi-instance)
        <li>openssh-client (only required for multi-instance)
        <li>openssh-server (only required for multi-instance)
        <li>libopenmpi-dev (only required for multi-instance)
        <li>horovod==0.27.0 (only required for multi-instance)
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit on Windows you will need:</p>
      <ul>
        <li><a href="/docs/general/Windows.md">Intel Model Zoo on Windows Systems prerequisites</a>
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After finishing the setup above, download the pretrained model based on `PRECISION` and set the
`PRETRAINED_MODEL` environment var to the path to the frozen graph.
If you run on Windows, please use a browser to download the pretrained model using the link below.
For Linux, run:
```
# FP32, FP16 and BFloat32 Pretrained model:
wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
export PRETRAINED_MODEL=$(pwd)/resnet50_v1.pb

# Int8 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50v1_5_int8_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/resnet50v1_5_int8_pretrained_model.pb

#BFloat16 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_v1_5_bfloat16.pb
export PRETRAINED_MODEL=$(pwd)/resnet50_v1_5_bfloat16.pb
```

Set the environment variables and run quickstart script on either Linux or Windows systems. See the [list of quickstart scripts](#quick-start-scripts) for details on the different options.

### Run on Linux
```
# cd to your model zoo directory
cd models

export PRETRAINED_MODEL=<path to the frozen graph downloaded above>
export DATASET_DIR=<path to the ImageNet TF records>
export PRECISION=<set the precision to "int8" or "fp32" or "bfloat16" or "fp16" or "bfloat32">
export OUTPUT_DIR=<path to the directory where log files and checkpoints will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/<script name>.sh
```

### Run on Windows
Using `cmd.exe`, run:
```
# cd to your model zoo directory
cd models

set PRETRAINED_MODEL=<path to the frozen graph downloaded above>
set DATASET_DIR=<path to the ImageNet TF records>
set PRECISION=<set the precision to "int8" or "fp32">
set OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
set BATCH_SIZE=<customized batch size value>

# Run a quickstart script (inference.sh and accuracy.sh are supported on windows) 
bash quickstart\image_recognition\tensorflow\resnet50v1_5\inference\cpu\<script name>.sh
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

* To run more advanced use cases, see the instructions for the available precisions [FP32](fp32/Advanced.md) [Int8](int8/Advanced.md) [BFloat16](bfloat16/Advanced.md) [FP16](fp16/Advanced.md) for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [Intel® Developer Catalog](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/resnet50v1-5-fp32-inference-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/resnet50v1-5-fp32-inference-tensorflow-container.html).
