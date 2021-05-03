<!--- 0. Title -->
# BERT Large FP32 training

<!-- 10. Description -->

This document has instructions for running
[BERT](https://github.com/google-research/bert#what-is-bert) FP32 training
using Intel-optimized TensorFlow.

For all fine-tuning the datasets (SQuAD, MultiNLI, MRPC etc..) and checkpoints
should be downloaded as mentioned in the [Google bert repo](https://github.com/google-research/bert).

Refer to google reference page for [checkpoints](https://github.com/google-research/bert#pre-trained-models).

<!--- 30. Datasets -->
## Datasets

### Pretrained models

Download and extract checkpoints the bert pretrained model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
The extracted directory should be set to the `CHECKPOINT_DIR` environment
variable when running the quickstart scripts.

For training from scratch, Wikipedia and BookCorpus need to be downloaded
and pre-processed.

### GLUE data

[GLUE data](https://gluebenchmark.com/tasks) is used when running BERT
classification training. Download and unpack the GLUE data by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e).

### SQuAD data

The Stanford Question Answering Dataset (SQuAD) dataset files can be downloaded
from the [Google bert repo](https://github.com/google-research/bert#squad-11).
The three files (`train-v1.1.json`, `dev-v1.1.json`, and `evaluate-v1.1.py`)
should be downloaded to the same directory. Set the `DATASET_DIR` to point to
that directory when running bert fine tuning using the SQuAD data.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_classifier_training.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/fp32_classifier_training.sh) | This script fine-tunes the bert base model on the Microsoft Research Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples. Download the [bert base pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [GLUE data](#glue-data). |
| [`fp32_squad_training.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/fp32_squad_training.sh) | This script fine-tunes bert using SQuAD data. Download the [bert large pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [squad data files](#squad-data). |
| [`fp32_squad_training_demo.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/fp32_squad_training_demo.sh) | This script does a short demo run of 0.01 epochs using SQuAD data. |

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
./quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/<script name>.sh
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
./quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/<script name>.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/bert-large-fp32-training-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/bert-large-fp32-training-tensorflow-container.html).

