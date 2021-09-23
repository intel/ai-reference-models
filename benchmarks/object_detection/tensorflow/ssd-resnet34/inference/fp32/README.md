<!--- 0. Title -->
# SSD-ResNet34 FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running [SSD-ResNet34](https://arxiv.org/pdf/1512.02325.pdf)
FP32 inference using Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

The SSD-ResNet34 accuracy scripts ([fp32_accuracy.sh](fp32_accuracy.sh)
and [fp32_accuracy_1200.sh](fp32_accuracy_1200.sh)) use the
[COCO validation dataset](http://cocodataset.org) in the TF records
format. See the [COCO dataset document](/datasets/coco/README.md) for
instructions on downloading and preprocessing the COCO validation dataset.

The performance benchmarking scripts ([fp32_inference.sh](fp32_inference.sh)
and [fp32_inference_1200.sh](fp32_inference_1200.sh)) use synthetic data,
so no dataset is required.



<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp32_accuracy_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/fp32/fp32_accuracy_1200.sh) | Runs an accuracy test using data in the TF records format with an input size of 1200x1200. |
| [fp32_accuracy.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/fp32/fp32_accuracy.sh) | Runs an accuracy test using data in the TF records format with an input size of 300x300. |
| [fp32_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/fp32/fp32_inference_1200.sh) | Runs inference with a batch size of 1 using synthetic data with an input size of 1200x1200. Prints out the time spent per batch and total samples/second. |
| [fp32_inference.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/fp32/fp32_inference.sh) | Runs inference with a batch size of 1 using synthetic data with an input size of 300x300. Prints out the time spent per batch and total samples/second. |
| [multi_instance_batch_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/fp32/multi_instance_batch_inference_1200.sh) | Uses numactl to run inference (batch_size=1) with one instance per socket. Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |
| [multi_instance_online_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/fp32/multi_instance_online_inference_1200.sh) | Uses numactl to run inference (batch_size=1) with 4 cores per instance. Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |

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
      <p>To run without AI Kit you will need:</p>
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
  </tr>
</table>

The [TensorFlow models](https://github.com/tensorflow/models) and
[benchmarks](https://github.com/tensorflow/benchmarks) repos are used by
SSD-ResNet34 FP32 inference. Clone those at the git SHAs specified
below and set the `TF_MODELS_DIR` environment variable to point to the
directory where the models repo was cloned.

```
git clone --single-branch https://github.com/tensorflow/models.git tf_models
git clone --single-branch https://github.com/tensorflow/benchmarks.git ssd-resnet-benchmarks
cd tf_models
export TF_MODELS_DIR=$(pwd)
git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
cd ../ssd-resnet-benchmarks
git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
cd ..
```

Download the SSD-ResNet34 pretrained model for either the 300x300 or 1200x1200
input size, depending on which [quickstart script](#quick-start-scripts) you are
going to run. Set the `PRETRAINED_MODEL` environment variable for the path to the
pretrained model that you'll be using.
```
# ssd-resnet34 300x300
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_fp32_bs1_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/ssd_resnet34_fp32_bs1_pretrained_model.pb

# ssd-resnet34 1200x1200
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
```

After installing the prerequisites and cloning the models and benchmarks
repos, and downloading the pretrained model, set environment variables
for the path to your `DATASET_DIR` (for accuracy testing only --
inference benchmarking uses synthetic data) and an `OUTPUT_DIR` where
log files will be written. Once the required environment variables are set,
you can then run a [quickstart script](#quick-start-scripts) from the
Model Zoo.
```
# cd to your model zoo directory
cd models

# set environment variables
export DATASET_DIR=<directory with the validation-*-of-* files (for accuracy testing only)>
export TF_MODELS_DIR=<path to the TensorFlow Models repo>
export PRETRAINED_MODEL=<path to the 300x300 or 1200x1200 pretrained model pb file>
export OUTPUT_DIR=<directory where log files will be written>

./quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/fp32/<script name>.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/ssd-resnet34-fp32-inference-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/ssd-resnet34-fp32-inference-tensorflow-container.html).

