<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training_convergence.sh` | Verifies training convergency (mini-convergency) for the specified precision (fp32, avx-fp32, or bf16). By default, uses `NUM_BATCH=50000`. |
| `training_performance.sh` | Verifies training performance for the specified precision (fp32, avx-fp32, or bf16). By default, uses `NUM_BATCH=10000`. |

> Note: The `avx-fp32` precision runs the same scripts as `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
