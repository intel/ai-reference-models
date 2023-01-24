<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`training_check_accuracy.sh`](/quickstart/recommendation/tensorflow/wide_deep_large_ds/training/cpu/training_check_accuracy.sh) | Trains the model for a specified number of steps (default is 500) and then compare the accuracy against the accuracy specified in the `TARGET_ACCURACY` env var (ex: `export TARGET_ACCURACY=0.75`). If the accuracy is not met, then script exits with error code 1. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |
| [`training.sh`](/quickstart/recommendation/tensorflow/wide_deep_large_ds/training/cpu/training.sh) | Trains the model for 10 epochs. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |
| [`training_demo.sh`](/quickstart/recommendation/tensorflow/wide_deep_large_ds/training/cpu/training_demo.sh) | A short demo run that trains the model for 100 steps. |
