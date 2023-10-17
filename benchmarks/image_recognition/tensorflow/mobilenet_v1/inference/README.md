<!--- 0. Title -->
# MobileNet V1 inference

<!-- 10. Description -->
## Description

This document has instructions for running MobileNet V1 inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running MobileNet V1.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs realtime inference using a default `batch_size=1` for the specified precision (int8, fp32, bfloat16). To run inference for throughtput, set `BATCH_SIZE` environment variable. |
| `inference_realtime_multi_instance.sh` | A multi-instance run that uses 4 cores per instance with `batch_size=1` for the specified precision (fp32, int8, bfloat16, bfloat32). Uses synthetic data if no DATASET_DIR is set|
| `inference_throughput_multi_instance.sh` | A multi-instance run that uses 4 cores per instance with `batch_size=448` for the specified precision (fp32, int8, bfloat16, bfloat32). Uses synthetic data if no DATASET_DIR is set |
| `accuracy.sh` | Measures the model accuracy (batch_size=100) for the specified precision (fp32, int8, bfloat16, bfloat32). |
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

After finishing the setup above, download the pretrained model based on `PRECISION`and set the
`PRETRAINED_MODEL` environment var to the path to the frozen graph.
If you run on Windows, please use a browser to download the pretrained model using the link below.
For Linux, run:
```
# FP32, BFloat16 and BFloat32 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/mobilenet_v1_1.0_224_frozen.pb
export PRETRAINED_MODEL=$(pwd)/mobilenet_v1_1.0_224_frozen.pb
```
```
# Int8 Pretrained model:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/mobilenetv1_int8_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/mobilenetv1_int8_pretrained_model.pb
```
Intel® Neural Compressor int8 quantized MobileNet V1 pre-trained model is available as another option to download and try.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/intel-neural-compressor/v1_13/mobilenetv1-inc-int8-inference.pb
export PRETRAINED_MODEL=$(pwd)/mobilenetv1-inc-int8-inference.pb
```
Check the [instructions](/quickstart/image_recognition/tensorflow/generate_int8/README.md) for more details on how to quantize FP32 model using Intel® Neural Compressor.

Set the environment variables and run quickstart script on either Linux or Windows systems. See the [list of quickstart scripts](#quick-start-scripts) for details on the different options.

### Run on Linux:
```
# cd to your AI Reference Models directory
cd models

export PRETRAINED_MODEL=<path to the frozen graph downloaded above>
export DATASET_DIR=<path to the ImageNet TF records>
export PRECISION=<set the precision to "int8" or "fp32" or "bfloat16" or "bfloat32">
export OUTPUT_DIR=<path to the directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/<script name>.sh
```

### Run on Windows
Using `cmd.exe` run:
```
# cd to your AI Reference Models directory
cd models

set PRETRAINED_MODEL=<path to the frozen graph downloaded above>
set DATASET_DIR=<path to the ImageNet TF records>
set PRECISION=<set the precision to "int8" or "fp32">
set OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
set BATCH_SIZE=<customized batch size value>
# Run a quick start script for inference or accuracy
bash quickstart\image_recognition\tensorflow\mobilenet_v1\inference\cpu\<script name>.sh
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

* To run more advanced use cases, see the instructions for the available precisions [FP32](fp32/Advanced.md) [Int8](int8/Advanced.md) [BFloat16](bfloat16/Advanced.md) for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [Intel® Developer Catalog](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)
  workload container:<br />
  [https://www.intel.com/content/www/us/en/developer/articles/containers/mobilenetv1-fp32-inference-tensorflow-container.html](https://www.intel.com/content/www/us/en/developer/articles/containers/mobilenetv1-fp32-inference-tensorflow-container.html).

