<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance with synthetic data for the specified precision (fp32, int8 or bf16). |
| `inference_throughput.sh` | Runs multi instance batch inference using 1 instance per socket with synthetic data for the specified precision (fp32, int8 or bf16). |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bf16). |
