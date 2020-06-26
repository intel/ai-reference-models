# BERT FP32 Training

This document has instructions for running
[BERT](https://github.com/google-research/bert#what-is-bert) FP32 training
using Intel-optimized TensorFlow.

For all fine-tuning the datasets (SQuAD, MultiNLI, MRPC etc..) and checkpoints
should be downloaded as mentioned in the [Google bert repo](https://github.com/google-research/bert).

Refer to google reference page for [checkpoints](https://github.com/google-research/bert#pre-trained-models).

## Example Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_training_multi_node.sh`](fp32_training_multi_node.sh) | This script is used by the Kubernetes pods to run training across multiple nodes using mpirun and horovod. |

These examples can be run the following environments:
* [Kubernetes](#kubernetes)

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
kubectl -k bert_large_fp32_training/examples/k8s/mlops/multi-node apply
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
kubectl -k bert_large_fp32_training/examples/k8s/mlops/single-node apply
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
kubectl -k bert_large_fp32_training/examples/k8s/mlops/multi-node delete
```

Remove the mpi-operator after running the example by running the following command with
the `mpi-operator.yaml` file that you originally deployed with:

```
kubectl delete -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.2.3/deploy/v1alpha2/mpi-operator.yaml
```
