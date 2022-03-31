<!--- 0. Title -->
# ResNet50 v1.5 FP32 training

<!-- 10. Description -->

This document has instructions for running ResNet50 v1.5 FP32 training using
Intel® Optimizations for TensorFlow* on Kubernetes*.



<!--- 20. Download link -->
## Download link

[resnet50v1-5-fp32-training-k8s.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50v1-5-fp32-training-k8s.tar.gz)

<!--- 30. Datasets -->
## Dataset

The ImageNet dataset is used in these ResNet50 v1.5 Kubernetes examples.
Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`launch_benchmark.py`](mlops/single-node/user-mounted-nfs/pod.yaml#L18) | Executes a short run using small batch sizes and a limited number of steps to demonstrate the training flow |

These quickstart scripts can be run in the following environment:
* [Kubernetes](#kubernetes)


<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the ResNet50 v1.5 FP32 training package:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50v1-5-fp32-training-k8s.tar.gz
tar -xvf resnet50v1-5-fp32-training-k8s.tar.gz
```

#### Prerequisites

Both single and multi-node deployments use [kustomize-v3.8.7](https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv3.8.7) to configure deployment parameters. Kustomize-v3.8.7 should be downloaded, extracted and moved to a directory within your PATH. You can verify that you've installed the correct version of kustomize by typing `kustomize version`. On MACOSX you would see:

```
{Version:kustomize/v3.8.7 GitCommit:ad092cc7a91c07fdf63a2e4b7f13fa588a39af4f BuildDate:2020-11-11T23:19:38Z GoOs:darwin GoArch:amd64}
```

### Execution

The Kubernetes* package for `ResNet50 v1.5 FP32 training` includes single and multi-node kubernetes deployments.
Within the single and multi-node deployments are common use cases that include storage and security 
variations that are common across different kubernetes installations. The directory tree within the model package is shown below, 
where single and multi-node directories are below the [mlops](https://en.wikipedia.org/wiki/MLOps) directory. 
Common use cases are found under the single and multi-node directories:

```
quickstart
└── mlops
      ├── multi-node
      │       ├── user-allocated-pvc
      │       └── user-mounted-nfs
      └── single-node
              ├── user-allocated-pvc
              └── user-mounted-nfs
```

#### Multi-node distributed training

The multi-node use cases (user-allocated-pvc, user-mounted-nfs) make the following assumptions:
- the [mpi-operator](/tools/k8s/devops/operators/mpi-operator.yaml) has been deployed on the cluster by [devops](https://en.wikipedia.org/wiki/DevOps) (see below).
- the OUTPUT_DIR parameter is a shared volume that is writable by the user and available cluster wide.
- the DATASET_DIR parameter is a dataset volume also available cluster wide (eg: using zfs or other performant storage). Typically these volumes are read-only.

##### [Devops](https://en.wikipedia.org/wiki/DevOps)

The k8 resources needed to run the multi-node resnet50v1-5-fp32-training-k8s quickstart require deployment of an mpi-operator.
See the [MPI Operator deployment](/k8s/common/tensorflow/KubernetesDevOps.md#mpi-operator-deployment) section of the Kubernetes DevOps document for instructions.

Once these resources have been deployed, the mlops user then has a choice 
of running resnet50v1-5-fp32-training-k8s multi-node (distributed training) or single-node. 

##### [Mlops](https://en.wikipedia.org/wiki/MLOps)

Distributed training is done by posting an MPIJob to the k8s api-server which is handled by the mpi-operator that was deployed by 
devops. The mpi-operator parses the MPIJob and then runs a launcher and workers specified in the MPIJob. Launcher and workers communicate through [horovod](https://github.com/horovod/horovod).
The distributed training algorithm is handled by [mpirun](https://www.open-mpi.org/doc/current/man1/mpirun.1.php). 

In a terminal, `cd` to the multi-node directory. Each use case under this directory has parameters that can be changed 
using kustomize's [cfg set](https://github.com/kubernetes-sigs/kustomize/blob/master/cmd/config/docs/commands/set.md)

##### [Mlops](https://en.wikipedia.org/wiki/MLOps)

###### User mounted nfs and user allocated pvc parameter values

|     NAME   |                 VALUE    |        DESCRIPTION          |
|------------|--------------------------|-----------------------------|
| DATASET_DIR| /datasets                | input dataset directory     |
| FS_ID      | 0                        | owner id of mounted volumes |
| GROUP_ID   | 0                        | process group id            |
| GROUP_NAME | root                     | process group name          |
| NFS_PATH   | /nfs                     | nfs path                    |
| NFS_SERVER | 0.0.0.0                  | nfs server                  |
| PVC_NAME   | workdisk                 | pvc name                    |
| PVC_PATH   | /pvc                     | pvc path                    |
| USER_ID    | 0                        | process owner id            |
| USER_NAME  | root                     | process owner name          |

For the user mounted nfs use case, the user should change NFS_PATH and NFS_SERVER.

For the user allocated pvc use case, the user should change PVC_NAME and PVC_PATH.

For example to change the NFS_SERVER address the user would run:

```
kustomize cfg set . NFS_SERVER <ip address> -R
```

To change the PVC_NAME the user would run:

```
kustomize cfg set . PVC_NAME <PVC Name> -R
```

In both use cases, the user should change the values below so the pod is deployed with the user's identity.

```
kustomize cfg set . FS_ID <Group ID> -R
kustomize cfg set . GROUP_ID <Group ID> -R
kustomize cfg set . GROUP_NAME <Group Name> -R
kustomize cfg set . USER_ID <User ID> -R
kustomize cfg set . USER_NAME <User Name> -R
```

The user should change the default namespace of all the resources by running the kustomize command:

```
pushd <use-case>
kustomize edit set namespace <User's namespace>
popd
```

This will place all resources within the specified namespace. Note: this namespace should be created prior to deployment.

The user can also change their default kubectl context by running

```
kubectl config set-context --current --namespace=<User's namespace>
```

Once the user has changed parameter values they can then deploy the use-case by running:

```
kustomize build  <use-case> > <use-case>.yaml
kubectl apply -f <use-case>.yaml
```

##### Multi-node training output

Viewing the log output of the resnet50v1_5 MPIJob is done by viewing the logs of the 
launcher pod. The launcher pod aggregrates output from the workerpods. 
This pod is found by filtering the list of pods for the name 'launcher'

```
kubectl get pods -oname|grep launch|cut -c5-
```

This can be combined with the kubectl logs subcommand to tail the output of the training job

```
kubectl logs -f $(kubectl get pods -oname|grep launch|cut -c5-)
```

Note that the mpirun parameter -output-filename is actually a directory and is set to $OUTPUT_DIR.

##### Multi-node training cleanup

Removing the mpijob and related resources is done by running:

```
kubectl delete -f <use-case>.yaml
```

#### Single-node training

Single node training is similar to the docker use case but the command is run within a pod.
Training is done by submitting a pod.yaml to the k8s api-server which results in the pod creation and running 
the fp32_training_demo.sh command within the pod's container.

In a terminal, `cd` to the single-node directory. Each use case under this directory has parameters that can be changed 
using kustomize's [cfg set](https://github.com/kubernetes-sigs/kustomize/blob/master/cmd/config/docs/commands/set.md)

##### [Mlops](https://en.wikipedia.org/wiki/MLOps)

###### User mounted nfs and user allocated pvc parameter values

|     NAME   |                 VALUE    |        DESCRIPTION          |
|------------|--------------------------|-----------------------------|
| DATASET_DIR| /datasets                | input dataset directory     |
| FS_ID      | 0                        | owner id of mounted volumes |
| GROUP_ID   | 0                        | process group id            |
| GROUP_NAME | root                     | process group name          |
| NFS_PATH   | /nfs                     | nfs path                    |
| NFS_SERVER | 0.0.0.0                  | nfs server                  |
| PVC_NAME   | workdisk                 | model-builder | pvc name    |
| PVC_PATH   | /pvc                     | model-builder | pvc path    |
| USER_ID    | 0                        | process owner id            |
| USER_NAME  | root                     | process owner name          |

For the user mounted nfs use case, the user should change NFS_PATH and NFS_SERVER.

For the user allocated pvc use case, the user should change PVC_NAME and PVC_PATH.

For example to change the NFS_SERVER address the user would run:

```
kustomize cfg set . NFS_SERVER <ip address> -R
```

To change the PVC_NAME the user would run:

```
kustomize cfg set . PVC_NAME <PVC Name> -R
```

In both use cases, the user should change the values below so the pod is deployed with the user's identity.

```
kustomize cfg set . FS_ID <Group ID> -R
kustomize cfg set . GROUP_ID <Group ID> -R
kustomize cfg set . GROUP_NAME <Group Name> -R
kustomize cfg set . USER_ID <User ID> -R
kustomize cfg set . USER_NAME <User Name> -R
```

The user should change the default namespace of all the resources by running the kustomize command:

```
pushd <use-case>
kustomize edit set namespace <User's namespace>
popd
```

This will place all resources within the specified namespace. Note: this namespace should be created prior to deployment.

The user can also change their default kubectl context by running

```
kubectl config set-context --current --namespace=<User's namespace>
```

Once the user has changed parameter values they can then deploy the use-case by running:

```
kustomize build  <use-case> > <use-case>.yaml
kubectl apply -f <use-case>.yaml
```

##### Single-node training output

Viewing the log output of the resnet50v1_5 Pod is done by viewing the logs of the 
training pod. This pod is found by filtering the list of pods for the name 'training'

```
kubectl get pods -oname|grep training|cut -c5-
```

This can be combined with the kubectl logs subcommand to tail the output of the training job

```
kubectl logs -f $(kubectl get pods -oname|grep training|cut -c5-)
```

##### Single-node training cleanup

Removing the pod and related resources is done by running:

```
kubectl delete -f <use-case>.yaml
```

<!--- 71. TroubleShooting -->
## TroubleShooting

- Pod doesn't start. Status is ErrImagePull.<br/>
  Docker recently implemented rate limits.<br/>
  See this [note](https://thenewstack.io/docker-hub-limits-what-they-are-and-how-to-route-around-them/) about rate limits and work-arounds.

- Argo workflow steps do not execute.<br/>
  Error from `argo get <workflow>` is 'failed to save outputs: Failed to establish pod watch: timed out waiting for the condition'.<br/>
  See this argo [issue](https://github.com/argoproj/argo/issues/4186). This is due to the workflow running as non-root.<br/>
  Devops will need to change the workflow-executor to k8sapi as described [here](https://github.com/argoproj/argo/blob/master/docs/workflow-executors.md).

- MpiOperator can't create workers. Error is '/bin/sh: /etc/hosts: Permission denied'. This is due to a bug in mpi-operator in the 'latest' container image
  when the workers run as non-root. See this [issue](https://github.com/kubeflow/mpi-operator/issues/288).<br/>
  Use the container images: mpioperator/mpi-operator:v02.3 and mpioperator/kubectl-delivery:v0.2.3.


<!--- 80. License -->
## License

[LICENSE](/LICENSE)

