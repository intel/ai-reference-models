<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_inference.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/fp32/fp32_inference.sh) | Runs inference on TF records and outputs performance metrics. |
| [`fp32_accuracy.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/fp32/fp32_accuracy.sh) | Processes the TF records to run inference and check accuracy on the results. |
| [`multi_instance_batch_inference.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/fp32/multi_instance_batch_inference.sh) | A multi-instance run that uses all the cores for each socket for each instance with a batch size of 448 and synthetic data. |
| [`multi_instance_online_inference.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/fp32/multi_instance_online_inference.sh) | A multi-instance run that uses 4 cores per instance with a batch size of 1. Uses synthetic data if no `DATASET_DIR` is set. |
