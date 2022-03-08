<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference (batch-size=1) using 4 cores per instance for the specified precision (fp32 or bfloat16). |
| `inference_throughput.sh` | Runs multi instance batch inference (batch-size=64 for the precisions fp32 or bfloat16, and batch-size=448 for int8 precision) using 1 instance per socket. |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32 or bfloat16). |
