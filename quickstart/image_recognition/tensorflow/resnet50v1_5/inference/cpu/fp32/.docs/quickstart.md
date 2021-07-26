<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/fp32/fp32_online_inference.sh) | Runs online inference (batch_size=1). |
| [`fp32_batch_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/fp32/fp32_batch_inference.sh) | Runs batch inference (batch_size=128). |
| [`fp32_accuracy.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/fp32/fp32_accuracy.sh) | Measures the model accuracy (batch_size=100). |
| [`multi_instance_batch_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/fp32/multi_instance_batch_inference.sh) | Uses numactl to run batch inference (batch_size=128) with one instance per socket for 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| [`multi_instance_online_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/fp32/multi_instance_online_inference.sh) | Uses numactl to run online inference (batch_size=1) using four cores per instance for 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
