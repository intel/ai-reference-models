<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](/quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/fp32/fp32_online_inference.sh) | Runs online inference (batch_size=1). |
| [`fp32_batch_inference.sh`](/quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/fp32/fp32_batch_inference.sh) | Runs batch inference (batch_size=100). |
| [`fp32_accuracy.sh`](/quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/fp32/fp32_accuracy.sh) | Measures the model accuracy (batch_size=100). |
| [`multi_instance_batch_inference.sh`](/quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/fp32/multi_instance_batch_inference.sh) | A multi-instance run that uses all the cores for each socket for each instance with a batch size of 56. Uses synthetic data if no `DATASET_DIR` is set. |
| [`multi_instance_online_inference.sh`](/quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/fp32/multi_instance_online_inference.sh) | A multi-instance run that uses 4 cores per instance with a batch size of 1. Uses synthetic data if no `DATASET_DIR` is set. |
