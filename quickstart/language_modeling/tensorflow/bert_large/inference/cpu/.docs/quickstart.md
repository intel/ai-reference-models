<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference for BERT large (SQuAD) using 4 cores per instance with batch size 1 ( for precisions: fp32, int8, bfloat16 and bfloat32) to compute latency. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_realtime_weight_sharing.sh` | Runs multi instance realtime inference with weight sharing for BERT large (SQuAD) using 4 cores per instance with batch size 1 ( for precisions: fp32, int8, bfloat16 and bfloat32) to compute latency for weight sharing. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput.sh` | Runs multi instance batch inference for BERT large (SQuAD) using 1 instance per socket with batch size 128 (for precisions: fp32, int8 or bfloat16) to compute throughput. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures BERT large (SQuAD) inference accuracy for the specified precision (fp32, int8 or bfloat16 and bfloat32). |
