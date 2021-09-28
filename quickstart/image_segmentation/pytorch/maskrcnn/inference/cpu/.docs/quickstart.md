<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, int8, avx-int8, or bf16). |
| `inference_throughput.sh` | Runs multi instance batch inference using 24 cores per instance for the specified precision (fp32, int8, avx-int8, or bf16). |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, int8, avx-int8, or bf16). |

> Note: The `avx-int8` precision runs the same scripts as `int8`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
