<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

<model name> in mlperf benchmark can run with full training or
fewer training steps. During training we can control if it will do the evaluation
or not.

| Script name | Description |
|-------------|-------------|
| [`bfloat16_training_demo.sh`](bfloat16_training_demo.sh) | Runs 100 training steps. The script runs in single instance mode by default, for multi instance mode set `MPI_NUM_PROCESSES`. |
| [`bfloat16_training.sh`](bfloat16_training.sh) | Runs 200 training steps, saves checkpoints and does evaluation. The script runs in single instance mode by default, for multi instance mode set `MPI_NUM_PROCESSES`. |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

