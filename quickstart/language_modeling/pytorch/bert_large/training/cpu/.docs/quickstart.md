<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_bert_pretrain_phase1.sh` | Runs BERT large pretraining phase 1 using max_seq_len=128 for the first 90% dataset for the specified precision (fp32, avx-fp32, or bf16). The script saves the model to the `OUTPUT_DIR` in a directory called `model_save`. |
| `run_bert_pretrain_phase2.sh` | Runs BERT large pretraining phase 2 using max_seq_len=512 with the remaining 10% of the dataset for the specified precision (fp32, avx-fp32, or bf16). Use path to the `model_save` directory from phase one as the `CHECKPOINT_DIR` for phase 2. |

> Note: The `avx-fp32` precision runs the same scripts as `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
