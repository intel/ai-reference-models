<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`bfloat16_training_demo.sh`](bfloat16_training_demo.sh) | Executes a demo run with a limited number of training steps to test performance. Set the number of steps using the `TRAIN_STEPS` environment variable (defaults to 100). |
| [`bfloat16_training.sh`](bfloat16_training.sh) | Runs multi-instance training to convergence. Download the backbone model specified in the instructions below and pass that directory path in the `BACKBONE_MODEL_DIR` environment variable. |
| [`bfloat16_training_accuracy.sh`](bfloat16_training_accuracy.sh) | Runs the model in eval mode to check accuracy. Specify which checkpoint files to use with the `CHECKPOINT_DIR` environment variable. |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
