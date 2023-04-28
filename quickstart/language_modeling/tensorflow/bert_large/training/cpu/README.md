<!--- 0. Title -->
# TensorFlow BERT Large Training

<!-- 10. Description -->

This section has instructions for running BERT Large Training with the SQuAD dataset.

Set the `OUTPUT_DIR` to point to the directory where all logs will get stored.
Set the `PRECISION` to choose the appropriate precision for training. Choose from `fp32`, `bfloat16`, or `fp16`

## Datasets

### SQuAD data
Download and unzip the BERT Large uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Set the `DATASET_DIR` to point to this directory when running BERT Large.
```
mkdir -p $DATASET_DIR && cd $DATASET_DIR
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training_squad.sh` | Uses mpirun to execute 1 process per socket for BERT Large training with the specified precision (fp32, bfloat16 or fp16). Logs for each instance are saved to the output directory. |



# TensorFlow BERT Large Pretraining

<!-- 10. Description -->

This section has instructions for running BERT Large Pretraining
using Intel-optimized TensorFlow.


<!--- 30. Datasets -->
## Datasets

### SQuAD data
Download and unzip the BERT Large uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Set the `DATASET_DIR` to point to this directory when running BERT Large.
```
mkdir -p $DATASET_DIR && cd $DATASET_DIR
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```

Follow [instructions to generate BERT pre-training dataset](https://github.com/IntelAI/models/blob/bert-lamb-pretraining-tf-2.2/quickstart/language_modeling/tensorflow/bert_large/training/bfloat16/HowToGenerateBERTPretrainingDataset.txt)
in TensorFlow record file format. The output TensorFlow record files are expected to be located in the dataset directory `${DATASET_DIR}/tf_records`. An example for the TF record file path should be
`${DATASET_DIR}/tf_records/part-00430-of-00500`.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `pretraining.sh` | Uses mpirun to execute 1 process per socket for BERT Large pretraining with the specified precision (fp32 or bfloat16). Logs for each instance are saved to the output directory. |

<!-- 60. Docker -->
## Docker

The BERT Large training model container includes the scripts and libraries
needed to run BERT Large fine tuning. To run one of the quickstart scripts
using this container, you'll need to provide volume mounts for the pretrained model,
dataset, and an output directory where log and checkpoint files will be written.
If switching between running squad and classifier training or running classifier training
multiple times, use a new empty `OUTPUT_DIR` to prevent incompatible checkpoints from getting picked up.

The snippet below shows a quickstart script running with a single instance:
```
export PRECISION=<specify the precision to run:fp32 and bfloat16>
CHECKPOINT_DIR=<path to the pretrained bert model directory>
DATASET_DIR=<path to the dataset being used>
OUTPUT_DIR=<directory where checkpoints and log files will be saved>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  intel/language-modeling:tf-latest-bert-large-training \
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
export PRECISION=<specify the precision to run:fp32, bfloat16 and bfloat32>
CHECKPOINT_DIR=<path to the pretrained bert model directory>
DATASET_DIR=<path to the dataset being used>
OUTPUT_DIR=<directory where checkpoints and log files will be saved>
MPI_NUM_PROCESSES=<number of sockets to use>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env PRECISION=&{PRECISION} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  intel/language-modeling:tf-latest-bert-large-training \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!-- 61. Advanced Options -->
### Advanced Options

See the [Advanced Options for Model Packages and Containers](/quickstart/common/tensorflow/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.
<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

