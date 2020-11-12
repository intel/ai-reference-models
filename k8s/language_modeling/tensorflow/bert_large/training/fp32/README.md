<!--- 0. Title -->
# BERT Large FP32 training

<!-- 10. Description -->

This document has instructions for running [BERT](https://github.com/google-research/bert#what-is-bert) FP32 training using
Intel® Optimizations for TensorFlow* on Kubernetes*.

For all fine-tuning the datasets (SQuAD, MultiNLI, MRPC etc..) and checkpoints
should be downloaded as mentioned in the [Google bert repo](https://github.com/google-research/bert).

Refer to google reference page for [checkpoints](https://github.com/google-research/bert#pre-trained-models).



<!--- 20. Download link -->
## Download link

[bert-large-fp32-training-k8s.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_1_0/bert-large-fp32-training-k8s.tar.gz)

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
| [`fp32_training_single_node.sh`](mlops/single-node/fp32_training_single_node.sh) | This script is used by the single node Kubernetes job to run bert classifier inference. |

These quickstart scripts can be run in the following environment:
* [Kubernetes](#kubernetes)


<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the BERT Large FP32 training package:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_1_0/bert-large-fp32-training-k8s.tar.gz
tar -xvf bert-large-fp32-training-k8s.tar.gz
```

### Execution

The Kubernetes* package for `BERT Large FP32 training` includes single and multi-node kubernetes deployments.
The directory tree within the kubernetes package is shown below, where single and multi-node directories are below the 
[mlops](https://en.wikipedia.org/wiki/MLOps) directory:

```
quickstart
└── mlops
    ├── multi-node
    └── single-node
```

#### prerequisites

Both single and multi-node deployments use [kustomize-v3.8.4](https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv3.8.4) to configure deployment parameters. This archive should be downloaded, extracted and the kustomize command should be moved to a directory within your PATH. You can verify the correct version of kustomize has been installed by typing `kustomize version`. On MACOSX you would see

```
{Version:kustomize/v3.8.4 GitCommit:8285af8cf11c0b202be533e02b88e114ad61c1a9 BuildDate:2020-09-19T15:39:21Z GoOs:darwin GoArch:amd64}
```


The kustomization files that the kustomize command references are located withing the following directories:

```
bert-large-fp32-training-k8s/quickstart/mlops/single-node/kustomization.yaml
bert-large-fp32-training-k8s/quickstart/mlops/multi-node/kustomization.yaml
```

#### multi-node distributed training

The multi-node use case makes the following assumptions:
- the mpi-operator has been deployed on the cluster by [devops](https://en.wikipedia.org/wiki/DevOps) (see below).
- the OUTPUT_DIR parameter is a nfs share volume that is writable and available cluster wide.
- the DATASET_DIR parameter is a dataset volume also available cluster wide (eg: using zfs or other performant storage).

##### [devops](https://en.wikipedia.org/wiki/DevOps)

The k8 resources needed to run the multi-node bert-large-fp32-training-k8s quickstart require deployment of the mpi-operator.
See the [MPI operator deployment](/k8s/common/tensorflow/KubernetesDevOps.md#mpi-operator-deployment) section of the Kubernetes DevOps document for instructions.

Once these resources have been deployed, the mlops user then has a choice 
of running bert-large-fp32-training-k8s multi-node (distributed training) or single-node. 

##### [mlops](https://en.wikipedia.org/wiki/MLOps)

Distributed training is done by posting an MPIJob to the k8s api-server which is handled by the mpi-operator that was deployed by 
devops. The mpi-operator parses the MPIJob and then runs a launcher and workers specified in the MPIJob. Launcher and workers communicate through horovod.
The distributed training algorithm is handled by mpirun. 

Make sure you are inside the multi-node directory:

```
cd bert-large-fp32-training-k8s/quickstart/mlops/multi-node
```

The parameters that can be changed within the MPIJob are shown in the table[^1] below:

|      NAME      |                VALUE                |    SET BY     |         DESCRIPTION         | COUNT | REQUIRED |
|----------------|-------------------------------------|---------------|-----------------------------|-------|----------|
| BERT_BASE_DIR  | bert_official/MRPC                  | model-builder | bert base dir               | 1     | Yes      |
| CHECKPOINT_DIR | /checkpoints                        | model-builder | checkpoint directory        | 1     | Yes      |
| DATASET_DIR    | /datasets                           | model-builder | input dataset directory     | 6     | Yes      |
| FS_ID          | 0                                   | model-builder | owner id of mounted volumes | 2     | Yes      |
| GLUE_DIR       | bert_official                       | model-builder | glue dir                    | 1     | Yes      |
| GROUP_ID       | 0                                   | model-builder | process group id            | 4     | Yes      |
| GROUP_NAME     | root                                | model-builder | process group name          | 2     | Yes      |
| IMAGE_SUFFIX   |                                     | model-builder | appended to image name      | 2     | No       |
| MODEL_DIR      | /workspace/bert-large-fp32-training | model-builder | container model directory   | 6     | No       |
| MODEL_NAME     | bert-large-fp32-training            | model-builder | name use-case               | 8     | No       |
| NFS_PATH       | /nfs                                | model-builder | nfs path                    | 6     | Yes      |
| NFS_SERVER     | 0.0.0.0                             | model-builder | nfs server                  | 2     | Yes      |
| OUTPUT_DIR     | output                              | model-builder | output dir basename         | 2     | Yes      |
| REGISTRY       | docker.io                           | model-builder | image location              | 2     | No       |
| USER_ID        | 0                                   | model-builder | process owner id            | 4     | Yes      |
| USER_NAME      | root                                | model-builder | process owner name          | 4     | Yes      |
| WORKERS        | 2                                   | model-builder | number of workers           | 1     | No       |

[^1]: The multi-node parameters table is generated by `kustomize cfg list-setters . --markdown`. See [list-setters](https://github.com/kubernetes-sigs/kustomize/blob/master/cmd/config/docs/commands/list-setters.md) for explanations of each column.

For example to change the NFS_SERVER IP address to 10.35.215.25 the user would run:

```
kustomize cfg set . NFS_SERVER 10.35.215.25
```
The required column that contains a 'Yes' indicates which values should be changed by the user.
The 'No' values indicate that the default values are fine. Note that the mlops user should run the
training process with their own uid/gid permissions by using kustomize to change the securityContext in the pod.yaml file.
This is done by running the following:

```
kustomize cfg set . FS_ID <Group ID>
kustomize cfg set . GROUP_ID <Group ID>
kustomize cfg set . GROUP_NAME <Group Name>
kustomize cfg set . USER_ID <User ID>
kustomize cfg set . USER_NAME <User Name>
```

Finally, the namespace can be changed by the user from the default namespace by running the kustomize command:

```
kustomize edit set namespace $USER
```

This will deploy the mpijob, mpi launcher and workers within the specified namespace. Note: this namespace should be created prior to deployment.
Once the user has changed parameter values they can then deploy the multi-node job by running:

```
kustomize build > multi-node.yaml
kubectl apply -f multi-node.yaml
```

##### multi-node training output

Viewing the log output of the bert_large MPIJob is done by viewing the logs of the 
launcher pod. The launcher pod aggregrates output from the workerpods. 
This pod is found by filtering the list of pods for the name 'launcher'

```
kubectl get pods -oname|grep launch|cut -c5-
```

This can be combined with the kubectl logs subcommand to tail the output of the training job

```
kubectl logs -f $(kubectl get pods -oname|grep launch|cut -c5-)
```

Note that the mpirun parameter -output-filename in openmpi-2.1.X causes a segfault when attempting to write to the $OUTPUT_DIR that is
NFS mounted when the securityContext has been changed to run as the user's UID/GID. Therefore do not add this parameter to the args within 
the mpi-job.yaml file.

##### multi-node training cleanup

Removing the mpijob and related resources is done by running:

```
kubectl delete -f multi-node.yaml
```

#### single-node training

Single node training is similar to the docker use case but the command is run within a pod.
Training is done by submitting a single-node.yaml to the k8s api-server which results in the pod creation and running 
the quickstart script within the pod's container.

Make sure you are inside the single-node directory:

```
cd bert-large-fp32-training-k8s/quickstart/mlops/single-node
```

The parameters that can be changed within the pod are shown in the table[^2] below:

|     NAME      |                VALUE                |    SET BY     |         DESCRIPTION         | COUNT | REQUIRED |
|---------------|-------------------------------------|---------------|-----------------------------|-------|----------|
| BERT_BASE_DIR | bert_official/MRPC                  | model-builder | name use-case               | 1     | No       |
| DATASET_DIR   | /datasets                           | model-builder | input dataset directory     | 3     | Yes      |
| FS_ID         | 0                                   | model-builder | owner id of mounted volumes | 1     | Yes      |
| GLUE_DIR      | bert_official                       | model-builder | name use-case               | 1     | No       |
| GROUP_ID      | 0                                   | model-builder | process group id            | 2     | Yes      |
| GROUP_NAME    | root                                | model-builder | process group name          | 1     | Yes      |
| IMAGE_SUFFIX  |                                     | model-builder | appended to image name      | 1     | No       |
| MODEL_DIR     | /workspace/bert-large-fp32-training | model-builder | container model directory   | 3     | No       |
| MODEL_NAME    | bert-large-fp32-training            | model-builder | name use-case               | 5     | No       |
| MODEL_SCRIPT  | fp32_training_single_node.sh        | model-builder | model script name           | 5     | No       |
| NFS_PATH      | /nfs                                | model-builder | nfs path                    | 3     | Yes      |
| NFS_SERVER    | 0.0.0.0                             | model-builder | nfs server                  | 1     | Yes      |
| OUTPUT_DIR    | output                              | model-builder | output dir basename         | 1     | Yes      |
| REGISTRY      | docker.io                           | model-builder | image location              | 1     | No       |
| USER_ID       | 0                                   | model-builder | process owner id            | 2     | Yes      |
| USER_NAME     | root                                | model-builder | process owner name          | 2     | Yes      |

[^2]: The single-node parameters table is generated by `kustomize cfg list-setters . --markdown`. See [list-setters](https://github.com/kubernetes-sigs/kustomize/blob/master/cmd/config/docs/commands/list-setters.md) for explanations of each column.

For example to change the NFS_SERVER IP address to 10.35.215.25 the user would run:

```
kustomize cfg set . NFS_SERVER 10.35.215.25
```

The required column that contains a 'Yes' indicates which values should be changed by the user.
The 'No' values indicate that the default values are fine. Note that the mlops user should run the
training process with their own uid/gid permissions by using kustomize to change the securityContext in the pod.yaml file.
This is done by running the following:

```
kustomize cfg set . FS_ID <Group ID>
kustomize cfg set . GROUP_ID <Group ID>
kustomize cfg set . GROUP_NAME <Group Name>
kustomize cfg set . USER_ID <User ID>
kustomize cfg set . USER_NAME <User Name>
```

Finally, the namespace can be changed by the user from the default namespace by running the kustomize command:

```
kustomize edit set namespace $USER
```

This will deploy the pod within the specified namespace. Note: this namespace should be created prior to deployment.
Once the user has changed parameter values they can then deploy the single-node pod by running:

```
kustomize build > single-node.yaml
kubectl apply -f single-node.yaml
```

##### single-node training output

Viewing the log output of the bert-large-fp32-training Pod is done by viewing the logs of the 
training pod. This pod is found by filtering the list of pods for the name 'training'

```
kubectl get pods -oname|grep training|cut -c5-
```

This can be combined with the kubectl logs subcommand to tail the output of the training job

```
kubectl logs -f $(kubectl get pods -oname|grep training|cut -c5-)
```

##### single-node training cleanup

Removing the pod and related resources is done by running:

```
kubectl delete -f single-node.yaml
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

