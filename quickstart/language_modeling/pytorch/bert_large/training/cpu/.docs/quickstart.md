<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `pretrain_phase1.sh` | Runs BERT large (SQuAD) pretraining phase 1 using max_seq_len=128 for the first 90% dataset for the specified precision (fp32 or bf16). The script saves the model to the `OUTPUT_DIR` in a directory called `model_save`. |
| `pretrain_phase2.sh` | Runs BERT large (SQuAD) pretraining phase 2 using max_seq_len=512 with the remaining 10% of the dataset for the specified precision (fp32 or bf16). Use path to the `model_save` directory from phase one as the `CHECKPOINT_DIR` for phase 2. |
