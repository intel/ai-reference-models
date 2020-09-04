<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the ResNet50 v1.5 FP32 training package:
```
wget <package url>
tar -xvf <package name>
```

### Execution

The model package for <model name> <precision> <mode> includes a deployment that does 'mlops' (machine learning operations) on kubernetes.
The directory tree within the model package is shown below:

```
quickstart
├── common
│   └── tensorflow
│       └── k8s
│           └── mlops
│               └── base
└── k8s
    └── mlops
        ├── multi-node
        └── single-node
```

The deployments use [kustomize](https://kustomize.io/) to configure deployment parameters. The kustomization files are located withing the 
following directories:

```
quickstart/common/tensorflow/k8s/mlops/base/kustomization.yaml
quickstart/k8s/mlops/single-node/kustomization.yaml
quickstart/k8s/mlops/multi-node/kustomization.yaml
```

The multi-node use case makes the following assumptions:
- the mpi-operator has been deployed on the cluster
- the nfs share is available cluster wide
- the model package has been extracted to a nfs shared volume
- the dataset volume is also available cluster wide (typically by mounting readonly and performant storage)

The parameters are configured by editing kustomize related files described in the sections below.

#### multi-node distributed training

##### devops

The k8 resources needed to run the multi-node resnet50v1-5 training quickstart require deployment of the mpi-operator.
See the MPI operator deployment section of the Kubernetes DevOps document
for instructions.

Once these resources have been deployed, the mlops user then has a choice 
of running resnet50v1-5 multi-node (distributed training) or single-node. 

##### mlops

Distributed training is done by posting an MPIJob to the k8s api-server which is handled by the mpi-operator that was deployed by 
the devops user. The mpi-operator then runs a 'launcher' and workers defined in the MPIJob which communicate through horovod.
The distributed training algorithm is handled by mpirun. 

The command to run an MPIJob is shown below:

```
kubectl -k <package dir>/quickstart/k8s/mlops/multi-node apply
```

Within the multi-node use case, a number of kustomize processing directives are enabled.

- An 'environment file' mlops.env that includes variables that will be injected as environment variables within the pod.
- The mlops user should change 'ROOT' to point to where the model package was extracted.

```
REGISTRY=docker.io
DATASET_DIR=/tf_dataset
MODEL_DIR=ROOT/<package dir>
OUTPUT_DIR=ROOT/<package dir>/output
```

The mlops user may run the distributed training job with their own uid/gid permissions by editing securityContext in the mlops-job-patch.yaml file.
The securityContext appears within Launcher and Worker sections. The runAsUser, runAsGroup and fsGroup should reflect the user's UID, GID.

```
securityContext:
  runAsUser: <User ID>
  runAsGroup: <Group ID>
  fsGroup: <Group ID>
```

#### multi-node distributed training output

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

#### single-node training

Single node training is similar to the docker use case but the command is run within a pod.
Training is done by submitting a pod.yaml to the k8s api-server which results in the pod creation and running 
the fp32_training_demo.sh command within the pod's container.

The command to run a pod is shown below:

```
kubectl -k <package dir>/quickstart/k8s/mlops/single-node apply
```

Within the single-node use case, the same number of kustomize processing directives are enabled as the multi-node.

- An 'environment file' mlops.env that includes variables that will be injected as environment variables within the pod.
- The mlops user should change 'ROOT' to point to where the model package was extracted.

```
REGISTRY=docker.io
DATASET_DIR=/tf_dataset
MODEL_DIR=ROOT/<package dir>
OUTPUT_DIR=ROOT/<package dir>/output
```

The mlops user may run the single-node training job with their own uid/gid permissions by editing securityContext in the pod-patch.yaml file.
The securityContext appears within Launcher and Worker sections. The runAsUser, runAsGroup and fsGroup should reflect the user's UID, GID.

```
securityContext:
  runAsUser: <User ID>
  runAsGroup: <Group ID>
  fsGroup: <Group ID>
```

#### single-node training output

Viewing the log output of the resnet50v1_5 Pod is done by viewing the logs of the 
training pod. This pod is found by filtering the list of pods for the name 'training'

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
kubectl -k <package dir>/quickstart/k8s/mlops/multi-node delete
```

Remove the mpi-operator after running the quickstart by running the following command with
the `mpi-operator.yaml` file that you originally deployed with:

```
kubectl delete -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.2.3/deploy/v1alpha2/mpi-operator.yaml
```

