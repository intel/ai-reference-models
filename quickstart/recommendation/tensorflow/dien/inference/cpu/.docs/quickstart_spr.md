<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, bfloat16 or bfloat32) with a default`batch_size=16`. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, bfloat16 or bfloat32) with a default `batch_size=65536`. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, bfloat16 or bfloat32) with a default `batch_size=128`. |
