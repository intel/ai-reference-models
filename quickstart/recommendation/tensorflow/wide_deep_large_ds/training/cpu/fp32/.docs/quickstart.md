<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_training_check_accuracy.sh`](fp32_training_check_accuracy.sh) | Trains the model for a specified number of steps (default is 500) and then compare the accuracy against the accuracy specified in the `TARGET_ACCURACY` env var (ex: `export TARGET_ACCURACY=0.75`). If the accuracy is not met, then script exits with error code 1. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |
| [`fp32_training.sh`](fp32_training.sh) | Trains the model for 10 epochs. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |
| [`fp32_training_demo.sh`](fp32_training_demo.sh) | A short demo run that trains the model for 100 steps. |
