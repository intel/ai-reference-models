<!--- 0. Title -->
# SSD-ResNet34 inference

<!-- 10. Description -->
## Description

This document has instructions for running SSD-ResNet34 inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

The SSD-ResNet34 accuracy script `accuracy.sh` uses the
[COCO validation dataset](http://cocodataset.org) in the TF records
format. See the [COCO dataset document](https://github.com/IntelAI/models/tree/master/datasets/coco) for
instructions on downloading and preprocessing the COCO validation dataset.
The inference scripts use synthetic data, so no dataset is required.

After the script to convert the raw images to the TF records file completes, rename the tf_records file:
```
mv ${OUTPUT_DIR}/coco_val.record ${OUTPUT_DIR}/validation-00000-of-00001
```
Set the `DATASET_DIR` to the folder that has the `validation-00000-of-00001`
file when running the accuracy test. Note that the inference performance
test uses synthetic dataset.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`accuracy_1200.sh`](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/accuracy_1200.sh) | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bfloat16) with an input size of 1200x1200. |
| [`accuracy.sh`](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/accuracy.sh) | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bfloat16) with an input size of 300x300. |
| [inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/inference_1200.sh) | Runs inference with a batch size of 1 using synthetic data for the specified precision (fp32, int8 or bfloat16) with an input size of 1200x1200. Prints out the time spent per batch and total samples/second. |
| [inference.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/inference.sh) | Runs inference with a batch size of 1 using synthetic data for the specified precision (fp32, int8 or bfloat16) with an input size of 300x300. Prints out the time spent per batch and total samples/second. |
| [multi_instance_online_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/multi_instance_online_inference_1200.sh) | Runs multi instance realtime inference (batch-size=1) using 4 cores per instance for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |
| [multi_instance_batch_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/fp32/multi_instance_batch_inference_1200.sh) | Runs multi instance batch inference (batch-size=16) using 1 instance per socket for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |




## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run SSD-ResNet34 inference in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. 
The dataset is required for the accuracy script.
By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# To test accuracy, also specify the dataset directory
export DATASET_DIR=<path to the dataset>
SCRIPT=accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

