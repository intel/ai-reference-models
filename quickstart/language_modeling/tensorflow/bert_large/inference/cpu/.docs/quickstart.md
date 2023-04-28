<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `profile.sh` | This script runs inference in profile mode with a default `batch_size=32`. |
| `inference.sh` | Runs realtime inference using a default `batch_size=1` for the specified precision (fp32 or bfloat16). To run inference for throughtput, set `BATCH_SIZE` environment variable. |
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference for BERT large (SQuAD) using 4 cores per instance with batch size 1 ( for precisions: fp32 and bfloat16) to compute latency. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_realtime_weightsharing.sh` | Runs multi instance realtime inference with weight sharing for BERT large (SQuAD) using 4 cores per instance with batch size 1 ( for precisions: fp32 and bfloat16) to compute latency for weight sharing. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference for BERT large (SQuAD) using 1 instance per socket with batch size 128 (for precisions: fp32 and bfloat16) to compute throughput. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures BERT large (SQuAD) inference accuracy for the specified precision (fp32 and bfloat16). |
