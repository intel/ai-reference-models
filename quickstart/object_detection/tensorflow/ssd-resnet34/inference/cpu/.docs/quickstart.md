<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference (batch-size=1) using 4 cores per instance for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput.sh` | Runs multi instance batch inference (batch-size=16) using 1 instance per socket for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bfloat16). |
