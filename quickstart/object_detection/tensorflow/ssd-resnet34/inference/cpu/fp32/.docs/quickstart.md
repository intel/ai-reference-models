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
