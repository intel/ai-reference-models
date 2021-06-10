<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [int8_accuracy.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/int8/int8_accuracy.sh) | Tests accuracy using the COCO dataset in the TF Records format with an input size of 300x300. |
| [int8_inference.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/int8/int8_inference.sh) | Run inference using synthetic data with an input size of 300x300 and outputs performance metrics. |
| [int8_accuracy_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/int8/int8_accuracy_1200.sh) | Tests accuracy using the COCO dataset in the TF Records format with an input size of 1200x1200. |
| [multi_instance_batch_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/int8/multi_instance_batch_inference_1200.sh) | Uses numactl to run inference (batch_size=1) with an input size of 1200x1200 and one instance per socket. Waits for all instances to complete, then prints a summarized throughput value. |
| [multi_instance_online_inference_1200.sh](/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/int8/multi_instance_online_inference_1200.sh) | Uses numactl to run inference (batch_size=1) with an input size of 1200x1200 and four cores per instance. Waits for all instances to complete, then prints a summarized throughput value. |
