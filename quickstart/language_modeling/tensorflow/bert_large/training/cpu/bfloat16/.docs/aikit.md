<!--- 50. AI Kit -->
## Run the model

From AI Kit, activate the TensorFlow language modeling environment:
```
conda activate tensorflow_language_modeling
```

If you are not using AI Kit you will need:
* Python 3
* [intel-tensorflow==2.4.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* git
* openmpi-bin (only required for multi-instance)
* openmpi-common (only required for multi-instance)
* openssh-client (only required for multi-instance)
* openssh-server (only required for multi-instance)
* libopenmpi-dev (only required for multi-instance)
* horovod==0.19.1 (only required for multi-instance)
* Clone the Model Zoo repo:
  ```
  git clone https://github.com/IntelAI/models.git
  ```

Next, set environment variables with paths to the [dataset](#datasets),
[checkpoint files](#pretrained-models), and an output directory,then run
a quickstart script. See [list of quickstart scripts](#quick-start-scripts)
for details on the different options.

The snippet below shows a quickstart script running with a single instance:
```
# cd to your model zoo directory
cd models

export CHECKPOINT_DIR=<path to the pretrained bert model directory>
export DATASET_DIR=<path to the dataset being used>
export OUTPUT_DIR=<directory where checkpoints and log files will be saved>

# Run a script for your desired usage
./quickstart/language_modeling/tensorflow/bert_large/training/cpu/bfloat16/<script name>.sh
```

To run distributed training (one MPI process per socket) for better throughput,
set the `MPI_NUM_PROCESSES` var to the number of sockets to use. Note that the
global batch size is mpi_num_processes * train_batch_size and sometimes the learning
rate needs to be adjusted for convergence. By default, the script uses square root
learning rate scaling.

For fine-tuning tasks like BERT, state-of-the-art accuracy can be achieved via
parallel training without synchronizing gradients between MPI workers. The
`mpi_workers_sync_gradients=[True/False]` var controls whether the MPI
workers sync gradients. By default it is set to "False" meaning the workers
are training independently and the best performing training results will be
picked in the end. To enable gradients synchronization, set the
`mpi_workers_sync_gradients` to true in BERT options. To modify the bert
options, modify the quickstart .sh script or call the `launch_benchmarks.py`
script directly with your preferred args.

The snippet below shows a quickstart script running with a multiple instances:
```
# cd to your model zoo directory
cd models

export CHECKPOINT_DIR=<path to the pretrained bert model directory>
export DATASET_DIR=<path to the dataset being used>
export OUTPUT_DIR=<directory where checkpoints and log files will be saved>
export MPI_NUM_PROCESSES=<number of sockets to use>

# Run a script for your desired usage
./quickstart/language_modeling/tensorflow/bert_large/training/cpu/bfloat16/<script name>.sh
```
