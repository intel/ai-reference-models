<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `download_dataset.sh` | Download and prepare the LibriSpeech inference dataset. See the [datasets section](#datasets) for instructions on it's usage. |
| `inference_realtime.sh` | Runs multi-instance inference using 4 cores per instance for the specified precision (fp32, avx-fp32, or bf16). |
| `inference_throughput.sh` | Runs multi-instance inference using 1 instance per socket for the specified precision (fp32, avx-fp32, or bf16). |
| `accuracy.sh` | Runs an inference accuracy test for the specified precision (fp32, avx-fp32, or bf16). |

> Note: The `avx-fp32` precision runs the same scripts as `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
