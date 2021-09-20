<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference for BERT large (SQuAD) using 4 cores per instance with batch size 1, and (350 steps and 50 warmup steps for precisions: fp32, int8), and (300 steps and 400 warmup steps for precisions: bfloat16). Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput.sh` | Runs multi instance batch inference for BERT large (SQuAD) using 1 instance per socket with 350 steps, 50 warmup steps and batch size 128 (for precisions: fp32, int8 or bfloat16). Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures BERT large (SQuAD) inference accuracy for the specified precision (fp32, int8 or bfloat16) with batch size 32. |
