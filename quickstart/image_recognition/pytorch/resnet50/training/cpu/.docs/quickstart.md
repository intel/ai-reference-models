<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training.sh` | Trains using one node for one epoch for the specified precision (fp32, avx-fp32, or bf16). |

> Note: The `avx-fp32` precision runs the same scripts as `fp32`,
> except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
