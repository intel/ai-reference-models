<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`training_demo.sh`](/quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/training_demo.sh) | Runs 100 training steps. The script runs in single instance mode by default, for multi instance mode set `MPI_NUM_PROCESSES`. |
| [`training.sh`](/quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/training.sh) | Uses mpirun to execute 2 processes with 1 process per socket with a batch size of 5120 for the specified precision (fp32 or bfloat16). Logs for each instance are saved to the output directory. |
