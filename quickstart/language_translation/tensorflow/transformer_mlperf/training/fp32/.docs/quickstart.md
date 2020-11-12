<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

<model name> in mlperf benchmark can run with full training or
fewer training steps. During training we can control if it will do the evaluation
or not.

| Script name | Description |
|-------------|-------------|
| [`fp32_training_demo.sh`](fp32_training_demo.sh) | Runs 100 training steps (run on a single socket of the CPU). |
| [`fp32_training.sh`](fp32_training.sh) | Runs 200 training steps, saves checkpoints and do evaluation (run on a single socket of the CPU). |
| [`fp32_training_mpirun.sh`](fp32_training_mpirun.sh) | Runs training in multi-instance mode "2 sockets in a single node for example" using mpirun for the specified number of processes. |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
