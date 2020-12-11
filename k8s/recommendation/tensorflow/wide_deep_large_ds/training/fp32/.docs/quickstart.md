<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_training_check_accuracy.sh`](mlops/pipeline/user-mounted-nfs/train_and_serve.yaml#L50) | Trains the model for a specified number of steps (default is 500) and then compare the accuracy against the specified target accuracy. If the accuracy is not met, then script exits with error code 1. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |
| [`launch_benchmark.py`](mlops/single-node/user-mounted-nfs/pod.yaml#L50) | Trains the model for 10 epochs if -- steps is not specified. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |

These quickstart scripts are run in the following environment:
* [Kubernetes](#kubernetes)
