<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [accuracy_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/accuracy_1200.sh) | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bfloat16) with an input size of 1200x1200. |
| [accuracy.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/accuracy.sh) | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bfloat16) with an input size of 300x300. |
| [inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/inference_1200.sh) | Runs inference with a batch size of 1 using synthetic data for the specified precision (fp32, int8 or bfloat16) with an input size of 1200x1200. Prints out the time spent per batch and total samples/second. |
| [inference.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/inference.sh) | Runs inference with a batch size of 1 using synthetic data for the specified precision (fp32, int8 or bfloat16) with an input size of 300x300. Prints out the time spent per batch and total samples/second. |
| [multi_instance_online_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/multi_instance_online_inference_1200.sh) | Runs multi instance realtime inference (batch-size=1) using 4 cores per instance for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |
| [multi_instance_batch_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/fp32/multi_instance_batch_inference_1200.sh) | Runs multi instance batch inference (batch-size=16) using 1 instance per socket for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |



