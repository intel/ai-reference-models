<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, int8 or bfloat16) with 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_realtime_weightsharing.sh` | Runs multi instance realtime inference with weight sharing for the specified precision (int8 or bfloat16) with 1500 steps and 100 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, int8 or bfloat16) with 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bfloat16). |
