<!--- 0. Title -->
# BERT Large BFloat16 training

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large BFloat16 training using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[bert-large-bfloat16-training.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/bert-large-bfloat16-training.tar.gz)

<!--- 30. Datasets -->
## Datasets

Follow instructions in [BERT Large datasets](/datasets/bert_data/README.md#training) to download and preprocess the dataset.
You can do either classification training or fine-tuning using SQuAD.


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`bfloat16_classifier_training.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/bfloat16/bfloat16_classifier_training.sh) | This script fine-tunes the bert base model on the Microsoft Research Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples. Download the [bert base uncased 12-layer, 768-hidden pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [GLUE data](#glue-data). |
| [`bfloat16_squad_training.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/bfloat16/bfloat16_squad_training.sh) | This script fine-tunes bert using SQuAD data. Download the [bert large uncased (whole word masking) pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [squad data files](#squad-data). |
| [`bfloat16_squad_training_demo.sh`](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/bfloat16/bfloat16_squad_training_demo.sh) | This script does a short demo run of 0.01 epochs using the `mini-dev-v1.1.json` file instead of the full SQuAD dataset. |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* git

Once the above dependencies have been installed, download and untar the model
package, set environment variables, and then run a quickstart script. See the
[datasets](#datasets) and [list of quickstart scripts](#quick-start-scripts) for more
details on the different options. If switching between running squad and classifier
training or running classifier training multiple times, use a new empty
`OUTPUT_DIR` to prevent incompatible checkpoints from getting picked up.

The snippet below shows a quickstart script running with a single instance:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/bert-large-bfloat16-training.tar.gz
tar -xvf bert-large-bfloat16-training.tar.gz
cd bert-large-bfloat16-training

CHECKPOINT_DIR=<path to the pretrained bert model directory>
DATASET_DIR=<path to the dataset being used>
OUTPUT_DIR=<directory where checkpoints and log files will be saved>

# Run a script for your desired usage
./quickstart/<script name>.sh
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

To run with multiple instances, these additional dependencies will need to be
installed in your environment:
* openmpi-bin
* openmpi-common
* openssh-client
* openssh-server
* libopenmpi-dev
* horovod==0.19.1

```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/bert-large-bfloat16-training.tar.gz
tar -xvf bert-large-bfloat16-training.tar.gz
cd bert-large-bfloat16-training

CHECKPOINT_DIR=<path to the pretrained bert model directory>
DATASET_DIR=<path to the dataset being used>
OUTPUT_DIR=<directory where checkpoints and log files will be saved>
MPI_NUM_PROCESSES=<number of sockets to use>

# Run a script for your desired usage
./quickstart/<script name>.sh
```



<!--- 60. Docker -->
## Docker

The BERT Large BFloat16 training model container includes the scripts and libraries
needed to run BERT Large BFloat16 fine tuning. To run one of the quickstart scripts
using this container, you'll need to provide volume mounts for the pretrained model,
dataset, and an output directory where log and checkpoint files will be written.
If switching between running squad and classifier training or running classifier
training multiple times, use a new empty `OUTPUT_DIR` to prevent incompatible
checkpoints from getting picked up.

The snippet below shows a quickstart script running with a single instance:
```
CHECKPOINT_DIR=<path to the pretrained bert model directory>
DATASET_DIR=<path to the dataset being used>
OUTPUT_DIR=<directory where checkpoints and log files will be saved>

docker run \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  intel/language-modeling:tf-latest-bert-large-bfloat16-training \
  /bin/bash quickstart/<script name>.sh
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
```
CHECKPOINT_DIR=<path to the pretrained bert model directory>
DATASET_DIR=<path to the dataset being used>
OUTPUT_DIR=<directory where checkpoints and log files will be saved>
MPI_NUM_PROCESSES=<number of sockets to use>

docker run \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  intel/language-modeling:tf-latest-bert-large-bfloat16-training \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

