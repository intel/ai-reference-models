<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`int8_online_inference.sh`](/quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/int8/int8_online_inference.sh) | Runs online inference (batch_size=1). |
| [`int8_batch_inference.sh`](/quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/int8/int8_batch_inference.sh) | Runs batch inference (batch_size=240). |
| [`int8_accuracy.sh`](/quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/int8/int8_accuracy.sh) | Measures the model accuracy (batch_size=100). |
| [`multi_instance_batch_inference.sh`](/quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/int8/multi_instance_batch_inference.sh) | A multi-instance run that uses all the cores for each socket for each instance with a batch size of 56. Uses synthetic data if no `DATASET_DIR` is set. |
| [`multi_instance_online_inference.sh`](/quickstart/image_recognition/tensorflow/mobilenet_v1/inference/cpu/int8/multi_instance_online_inference.sh) | A multi-instance run that uses 4 cores per instance with a batch size of 1. Uses synthetic data if no `DATASET_DIR` is set. |
