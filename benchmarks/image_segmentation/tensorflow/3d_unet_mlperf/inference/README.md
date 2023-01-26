<!--- 0. Title -->
# TensorFlow MLPerf 3D U-Net inference

<!-- 10. Description -->
## Description

This document has instructions for running MLPerf 3D U-Net inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

Download [Brats 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) separately and unzip the dataset.

Set the `DATASET_DIR` to point to the directory that contains the dataset files when running MLPerf 3D U-Net accuracy script.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference.sh` | Runs realtime inference using a default `batch_size=1` for the specified precision (int8, fp32 or bfloat16). To run inference for throughtput, set `BATCH_SIZE` environment variable. |
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (int8, fp32 or bfloat16) with 100 steps and 50 warmup steps. Dummy data is used for performance evaluation. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (int8, fp32 or bfloat16) with 100 steps and 50 warmup steps. Dummy data is used for performance evaluation. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (int8, fp32 or bfloat16). |

<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md) on Linux or Windows systems.

<table>
  <tr>
    <th>Setup using AI Kit</th>
    <th>Setup without AI Kit</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Kit on Linux you will need:</p>
      <ul>
        <li>Activate the tensorflow conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>git
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

Download the pre-trained model based on precision:
* Download the [FP32 and BFloat16 pre-trained model](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/3dunet_dynamic_ndhwc.pb). 3D-Unet BFloat16 inference depends on Auto-Mixed_precision to convert graph from FP32 to BFloat16 online.
* Download the [Int8 pre-trained model](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/3dunet_int8_fully_quantized_perchannel.pb).

In this example, we are using the model, trained using the fold 1 BRATS 2019 data.
The validation files have been copied from [here](https://github.com/mlcommons/inference/tree/r0.7/vision/medical_imaging/3d-unet/folds).

Set the the `PRETRAINED_MODEL` environment variable to point to where the pre-trained model file was downloaded

### Run on Linux
Install dependencies:
```
# install numactl
pip install numactl

# install the model dependencies in requirements.txt if you would run accuracy.sh
pip install -r models/benchmarks/image_segmentation/tensorflow/3d_unet_mlperf/requirements.txt
```

Set the environment variables and one of the
[quickstart scripts](#quick-start-scripts). Currently, for performance evaluation dummy data is used.
Set DATASET_DIR if you run `accuracy.sh` to calculate the model accuracy.

```
# navigate to your Model Zoo directory
cd models

export DATASET_DIR=<path to the dataset directory>
export PRECISION=<set the precision "fp32", "int8" or "bfloat16">
export OUTPUT_DIR=<path to the directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model file based on the chosen precision>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# run a script for (example inference_realtime_multi_instance.sh)
./quickstart/image_segmentation/tensorflow/3d_unet_mlperf/inference/cpu/inference_realtime_multi_instance.sh
```

### Run on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).

Set the environment variables and run `inference.sh` from the
[quickstart script](#quick-start-scripts). Currently, for performance evaluation dummy data is used.
```
# navigate to your Model Zoo directory
cd models

set PRECISION=<set the precision "fp32" or "bfloat16">
set OUTPUT_DIR=<path to the directory where log files will be written>
set PRETRAINED_MODEL=<path to the pretrained model file based on the chosen precision>
# Set the BATCH_SIZE, or the script will use a default value BATCH_SIZE="1".
set BATCH_SIZE=<customized batch size value>

# run a script for inference
bash quickstart\image_segmentation\tensorflow\3d_unet_mlperf\inference\cpu\inference.sh
```
> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables.
As an example, if the output folder location is `D:\user\output`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\output
> /d/user/output
>```
>Then, set the `OUTPUT_DIR` environment variable `set OUTPUT_DIR=/d/user/output`.

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

