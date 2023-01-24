<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`inference.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference.sh) | Runs inference and outputs performance metrics. Uses synthetic data if no `DATASET_DIR` is set. Supported versions are (fp32, int8, bfloat16) |
| [`accuracy.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/accuracy.sh) | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required). Supported versions are (fp32, int8, bfloat16, bfloat32). |
| [`inference_throughput_multi_instance.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference_throughput_multi_instance.sh) | A multi-instance run that uses all the cores for each socket for each instance with a batch size of 448 and synthetic data. Supported versions are (fp32, int8, bfloat16, bfloat32) |
| [`inference_realtime_multi_instance.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference_realtime_multi_instance.sh) | A multi-instance run that uses 4 cores per instance with a batch size of 1. Uses synthetic data if no `DATASET_DIR` is set. Supported versions are (fp32, int8, bfloat16, bfloat32) |
