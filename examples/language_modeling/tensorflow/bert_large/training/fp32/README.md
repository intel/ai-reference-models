# BERT FP32 Training

This document has instructions for running
[BERT](https://github.com/google-research/bert#what-is-bert) FP32 training
using Intel-optimized TensorFlow.

For all fine-tuning the datasets (SQuAD, MultiNLI, MRPC etc..) and checkpoints
should be downloaded as mentioned in the [Google bert repo](https://github.com/google-research/bert).

Refer to google reference page for [checkpoints](https://github.com/google-research/bert#pre-trained-models).

## Datasets

### Pretrained models

Download and extract checkpoints the bert pretrained model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
The extracted directory should be set to the `CHECKPOINT_DIR` environment
variable when running example scripts.

For training from scratch, Wikipedia and BookCorpus need to be downloaded
and pre-processed.

### GLUE data

[GLUE data](https://gluebenchmark.com/tasks) is used when running BERT
classification training. Download and unpack the GLUE data by running
(this script)[https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e].

### SQuAD data

The Stanford Question Answering Dataset (SQuAD) dataset files can be downloaded
from the [Google bert repo](https://github.com/google-research/bert#squad-11).
The three files (`train-v1.1.json`, `dev-v1.1.json`, and `evaluate-v1.1.py`)
should be downloaded to the same directory. Set the `DATASET_DIR` to point to
that directory when running bert fine tuning using the SQuAD data.

## Example Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_classifier_training.sh`](fp32_classifier_training.sh) | This script fine-tunes the bert base model on the Microsoft Research Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples. Download the [bert base pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [GLUE data](#glue-data). |
| [`fp32_squad_training.sh`](fp32_squad_training.sh) | This script fine-tunes bert using SQuAD data. Download the [bert large pretrained model](https://github.com/google-research/bert#pre-trained-models) and set the `CHECKPOINT_DIR` to that directory. The `DATASET_DIR` should point to the [squad data files](#squad-data). |
| [`fp32_training_single_node.sh`](fp32_training_single_node.sh) | This script is used by the single node Kubernetes job to run bert classifier inference. |
| [`fp32_training_multi_node.sh`](fp32_training_multi_node.sh) | This script is used by the Kubernetes pods to run bert classifier training across multiple nodes using mpirun and horovod. |

These examples can be run the following environments:
* [Bare metal](#bare-metal)
* [Docker](#docker)
* [Kubernetes](#kubernetes)

## Bare Metal

To run on bare metal, the following prerequisites must be installed in your enviornment:
* Python 3
* [intel-tensorflow==2.1.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* git

Once the above dependencies have been installed, download and untar the model
package, set environment variables, and then run an example script. See the
[datasets](#datasets) and [list of example scripts](#example-scripts) for more
details on the different options.

The snippet below shows an example running with a single instance:
```
wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/bert-large-fp32-training.tar.gz
tar -xvf bert-large-fp32-training.tar.gz
cd bert-large-fp32-training

CHECKPOINT_DIR=<path to the pretrained bert model directory>
DATASET_DIR=<path to the dataset being used>
OUTPUT_DIR=<directory where checkpoints and log files will be saved>

# Run a script for your desired usage
./examples/<script name>.sh
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
options, modify the example .sh script or call the `launch_benchmarks.py`
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
wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/bert-large-fp32-training.tar.gz
tar -xvf bert-large-fp32-training.tar.gz
cd bert-large-fp32-training

CHECKPOINT_DIR=<path to the pretrained bert model directory>
DATASET_DIR=<path to the dataset being used>
OUTPUT_DIR=<directory where checkpoints and log files will be saved>
MPI_NUM_PROCESSES=<number of sockets to use>

# Run a script for your desired usage
./examples/<script name>.sh
```

## Docker

The bert FP32 training model container includes the scripts and libraries
needed to run bert large FP32 fine tuning. To run one of the example usage scripts
using this container, you'll need to provide volume mounts for the pretrained model,
dataset, and an output directory where log and checkpoint files will be written.

The snippet below shows an example running with a single instance:
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
  amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-language-modeling-bert-large-fp32-training \
  /bin/bash examples/<script name>.sh
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
options, modify the example .sh script or call the `launch_benchmarks.py`
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
  amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-language-modeling-bert-large-fp32-training \
  /bin/bash examples/<script name>.sh
```

## Kubernetes

Download and untar the bert large FP32 training package:
```
wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/bert-large-fp32-training.tar.gz
tar -xvf bert-large-fp32-training.tar.gz
```

### Execution

The model package for bert large FP32 training includes a deployment that does 'mlops' (machine learning operations) on kubernetes.
The directory tree within the model package is shown below:

```
examples
└── k8s
    └── mlops
        ├── base
        ├── multi-node
        └── single-node
```

The deployments use [kustomize](https://kustomize.io/) to configure deployment parameters. The kustomization files are located withing the
following directories:

```
examples/k8s/mlops/single-node/kustomization.yaml
examples/k8s/mlops/base/kustomization.yaml
examples/k8s/mlops/multi-node/kustomization.yaml
```

The multi-node use case makes the following assumptions:
- the mpi-operator has been deployed on the cluster
- the nfs share is available cluster wide
- the model package has been extracted to a nfs shared volume
- the dataset volume is also available cluster wide (typically by mounting readonly and performant storage)

The parameters are configured by editing kustomize related files described in the sections below.

#### multi-node distributed training

##### devops

The k8 resources needed to run the multi-node training example require deployment of the mpi-operator.
See the MPI operator deployment section of the Kubernetes DevOps document
for instructions.

Once these resources have been deployed, the mlops user then has a choice
of running multi-node (distributed training) or single-node.

##### mlops

Distributed training is done by posting an MPIJob to the k8s api-server which is handled by the mpi-operator that was deployed by
the devops user. The mpi-operator then runs a 'launcher' and workers defined in the MPIJob which communicate through horovod.
The distributed training algorithm is handled by mpirun.

The command to run an MPIJob is shown below:

```
kubectl -k bert-large-fp32-training/examples/k8s/mlops/multi-node apply
```

Within the multi-node use case, a number of kustomize processing directives are enabled.

- An 'environment file' mlops.env that includes variables that will be injected as environment variables within the pod.
- The mlops user should change 'ROOT' to point to where the model package was extracted.

```
REGISTRY=docker.io
DATASET_DIR=/tf_dataset
WORKSPACE=/workspace
MODEL_DIR=bert_larg_fp32_training
OUTPUT_DIR=/tmp/output
```

The mlops user may run the distributed training job with their own uid/gid permissions by editing securityContext in the mlops-job-patch.yaml file.
The securityContext appears within Launcher and Worker sections. The runAsUser, runAsGroup and fsGroup should reflect the user's UID, GID at the end.

```
securityContext:
  runAsUser: <User ID>
  runAsGroup: <Group ID>
  fsGroup: <Group ID>
```

#### multi-node distributed training output

Viewing the log output of the bert large MPIJob is done by viewing the logs of the
launcher pod. The launcher pod aggregrates output from the workerpods.
This pod is found by filtering the list of pods for the name 'launcher'

```
kubectl get pods -oname|grep launch|cut -c5-
```

This can be combined with the kubectl logs subcommand to tail the output of the training job

```
kubectl logs -f $(kubectl get pods -oname|grep launch|cut -c5-)
```

#### single-node training

Single node training is similar to the docker use case but the command is run within a pod.
Training is done by submitting a pod.yaml to the k8s api-server which results in the pod creation and running
the /workspace/bert-large-fp32-training/examples/fp32_training_single_node.sh command within the pod's container.

The command to run a pod is shown below:

```
kubectl -k bert-large-fp32-training/examples/k8s/mlops/single-node apply
```

Within the single-node use case, the same number of kustomize processing directives are enabled as the multi-node.

- An 'environment file' mlops.env that includes variables that will be injected as environment variables within the pod.
- The mlops user should change 'ROOT' to point to where the model package was extracted.

```
REGISTRY=docker.io
DATASET_DIR=/tf_dataset
WORKSPACE=/workspace
MODEL_DIR=bert_larg_fp32_training
OUTPUT_DIR=/tmp/output
```

The mlops user may run the single-node training job with their own uid/gid permissions by editing securityContext in the pod-patch.yaml file.
The securityContext appears within Launcher and Worker sections. The runAsUser, runAsGroup and fsGroup should reflect the user's UID, GID at the end.

```
securityContext:
  runAsUser: <User ID>
  runAsGroup: <Group ID>
  fsGroup: <Group ID>
```

#### single-node training output

Viewing the log output of the bert large
training pod is done by filtering the list of pods for the name 'training'.

```
kubectl get pods -oname|grep training|cut -c5-
```

This can be combined with the kubectl logs subcommand to tail the output of the training job

```
kubectl logs -f $(kubectl get pods -oname|grep training|cut -c5-)
```

#### Cleanup

Removing this MPIJob (and stopping training) is done by running:

```
kubectl -k bert-large-fp32-training/examples/k8s/mlops/multi-node delete
```

Remove the mpi-operator after running the example by running the following command with
the `mpi-operator.yaml` file that you originally deployed with:

```
kubectl delete -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.2.3/deploy/v1alpha2/mpi-operator.yaml
```
