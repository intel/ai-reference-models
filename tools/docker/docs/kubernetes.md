<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the <model name> package:
```
wget <package url>
tar -xvf <package name>
```

### Execution

The model package for <model name> includes a deployment that does 'mlops' (machine learning operations) on kubernetes.
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

The deployments use [kustomize](https://kustomize.io/) to configure deployment parameters. The kustomization files are located within the
following directories:

```
quickstart/common/tensorflow/k8s/mlops/base/kustomization.yaml
quickstart/k8s/mlops/single-node/kustomization.yaml
quickstart/k8s/mlops/multi-node/kustomization.yaml
```

The parameters are configured by editing kustomize related files described in the sections below.

#### multi-node distributed <mode>

##### devops

The k8 resources needed to run the multi-node <model name> quickstart require deployment of the <devops component>

Once these resources have been deployed, the mlops user then has a choice
of running <model name> multi-node (distributed <mode>) or single-node.

##### mlops

The command to run multi-node is shown below:

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

The mlops user may run the distributed <mode> job with their own uid/gid permissions by editing securityContext in the mlops-job-patch.yaml file.
The securityContext appears within Launcher and Worker sections. The runAsUser, runAsGroup and fsGroup should reflect the user's UID, GID.

```
securityContext:
  runAsUser: <User ID>
  runAsGroup: <Group ID>
  fsGroup: <Group ID>
```

#### single-node <mode>

Single node <mode> is similar to the docker use case but the command is run within a pod.
Training is done by submitting a pod.yaml to the k8s api-server which results in the pod creation and running
the fp32_<mode>_demo.sh command within the pod's container.

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

The mlops user may run the single-node <mode> job with their own uid/gid permissions by editing securityContext in the pod-patch.yaml file.
The securityContext appears within Launcher and Worker sections. The runAsUser, runAsGroup and fsGroup should reflect the user's UID, GID.

```
securityContext:
  runAsUser: <User ID>
  runAsGroup: <Group ID>
  fsGroup: <Group ID>
```

#### Cleanup

##### multi-node

```
kubectl -k <package dir>/quickstart/k8s/mlops/single-node delete
```

##### single-node

```
kubectl -k <package dir>/quickstart/k8s/mlops/multi-node delete
```
