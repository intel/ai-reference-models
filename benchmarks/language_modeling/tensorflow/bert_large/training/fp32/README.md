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

Follow instructions in [BERT Large datasets](/datasets/bert_data/README.md#training) to download and preprocess the dataset.
You can do either classification training or fine-tuning using SQuAD.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_classifier_training.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/fp32_classifier_training.sh) | This script fine-tunes the bert base model on the Microsoft Research Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples. Download the [bert base uncased 12-layer, 768-hidden pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [GLUE data](#glue-data). |
| [`fp32_squad_training.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/fp32_squad_training.sh) | This script fine-tunes bert using SQuAD data. Download the [bert large uncased (whole word masking) pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [squad data files](#squad-data). |
| [`fp32_squad_training_demo.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/fp32_squad_training_demo.sh) | This script does a short demo run of 0.01 epochs using the `mini-dev-v1.1.json` file instead of the full SQuAD dataset. |

<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit</th>
    <th>Setup without AI Kit</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Kit you will need:</p>
      <ul>
        <li>numactl
        <li>unzip
        <li>wget
        <li>openmpi-bin (only required for multi-instance)
        <li>openmpi-common (only required for multi-instance)
        <li>openssh-client (only required for multi-instance)
        <li>openssh-server (only required for multi-instance)
        <li>libopenmpi-dev (only required for multi-instance)
        <li>horovod==0.21.0 (only required for multi-instance)
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li>intel-tensorflow>=2.5.0
        <li>git
        <li>numactl
        <li>openmpi-bin (only required for multi-instance)
        <li>openmpi-common (only required for multi-instance)
        <li>openssh-client (only required for multi-instance)
        <li>openssh-server (only required for multi-instance)
        <li>libopenmpi-dev (only required for multi-instance)
        <li>horovod==0.21.0 (only required for multi-instance)
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After your setup is done, export environment variables with paths to the [dataset](#datasets),
[checkpoint files](#pretrained-models), and an output directory, then run
a quickstart script. If switching between running squad and classifier training
or running classifier training multiple times, use a new empty `OUTPUT_DIR` to
prevent incompatible checkpoints from getting picked up. See the
[list of quickstart scripts](#quick-start-scripts) for details on the different options.

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

The snippet below shows a quickstart script running with multiple instances:
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

