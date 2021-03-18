<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`multi_instance_training_demo.sh`](multi_instance_training_demo.sh) | Uses mpirun to execute 2 processes with 1 process per socket with a batch size of 256 for 50 steps. |
| [`multi_instance_training.sh`](multi_instance_training.sh) | Uses mpirun to execute 2 processes with 1 process per socket with a batch size of 256. Checkpoint files and logs for each instance are saved to the output directory. Note that this will take a considerable amount of time. |
| [`fp32_training_demo.sh`](fp32_training_demo.sh) | Executes a short run using small batch sizes and a limited number of steps to demonstrate the training flow |
| [`fp32_training_1_epoch.sh`](fp32_training_1_epoch.sh) | Executes a test run that trains the model for 1 epoch and saves checkpoint files to an output directory. |
| [`fp32_training_full.sh`](fp32_training_full.sh) | Trains the model using the full dataset and runs until convergence (90 epochs) and saves checkpoint files to an output directory. Note that this will take a considerable amount of time. |
| [`multi_instance_training_demo.sh`](multi_instance_training_demo.sh) | Uses numactl to execute one instance per socket of a short run using small batch sizes and a limited number of steps to demonstrate the training flow |
| [`multi_instance_training.sh`](multi_instance_training.sh) | Uses numactl to execute one instance per socket for the full training flow. Checkpoint files and logs for each instance are saved to the output directory. Note that this will take a considerable amount of time. |


These quick start scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

