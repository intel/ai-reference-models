<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`bfloat16_training_demo.sh`](bfloat16_training_demo.sh) | Executes a short run using small batch sizes and a limited number of steps to demonstrate the training flow |
| [`bfloat16_training_1_epoch.sh`](bfloat16_training_1_epoch.sh) | Executes a test run that trains the model for 1 epoch and saves checkpoint files to an output directory. |
| [`bfloat16_training_full.sh`](bfloat16_training_full.sh) | Trains the model using the full dataset and runs until convergence (90 epochs) and saves checkpoint files to an output directory. Note that this will take a considerable amount of time. |

These quick start scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

