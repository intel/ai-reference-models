<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference (batch-size=1) using 4 cores per instance for the specified precision (fp32 or bfloat16). |
| `inference_throughput.sh` | Runs multi instance batch inference (batch-size=64) using 1 instance per socket for the specified precision (fp32 or bfloat16). |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32 or bfloat16). |
