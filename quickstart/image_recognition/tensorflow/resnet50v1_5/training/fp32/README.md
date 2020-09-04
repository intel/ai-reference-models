<!--- 0. Title -->
# ResNet50 v1.5 FP32 Training

<!-- 10. Description -->

This document has instructions for running ResNet50 v1.5 FP32 training
using Intel-optimized TensorFlow.

Note that the ImageNet dataset is used in these ResNet50 v1.5 examples. To download and preprocess
the ImageNet dataset, see the [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
from the TensorFlow models repo.


<!--- 20. Download link -->
## Download link

[resnet50v1-5-fp32-training.tar.gz](https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/resnet50v1-5-fp32-training.tar.gz)

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_training_demo.sh`](fp32_training_demo.sh) | Executes a short run using small batch sizes and a limited number of steps to demonstrate the training flow |
| [`fp32_training_1_epoch.sh`](fp32_training_1_epoch.sh) | Executes a test run that trains the model for 1 epoch and saves checkpoint files to an output directory. |
| [`fp32_training_full.sh`](fp32_training_full.sh) | Trains the model using the full dataset and runs until convergence (90 epochs) and saves checkpoint files to an output directory. Note that this will take a considerable amount of time. |

These quick start scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
* [Kubernetes](#kubernetes)


<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your enviornment:
* Python 3
* [intel-tensorflow==2.1.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package and then run a [quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where checkpoint and log files will be written>

wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/resnet50v1-5-fp32-training.tar.gz
tar -xvf resnet50v1_5_fp32_training.tar.gz
cd resnet50v1_5_fp32_training

quickstart/<script name>.sh
```


!<--- 60. Docker -->
## Docker

The ResNet50 v1.5 FP32 training model container includes the scripts
and libraries needed to run ResNet50 v1.5 FP32 training. To run one of the model
training quickstart scripts using this container, you'll need to provide volume mounts for
the ImageNet dataset and an output directory where checkpoint files will be written.

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where checkpoint and log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-image-recognition-resnet50v1-5-fp32-training \
  /bin/bash quickstart/<script name>.sh
```


<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the ResNet50 v1.5 FP32 training package:
```
wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/resnet50v1-5-fp32-training.tar.gz
tar -xvf resnet50v1-5-fp32-training.tar.gz
```

### Execution

The model package for ResNet50 v1.5 FP32 training includes a deployment that does 'mlops' (machine learning operations) on kubernetes.
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
kubectl -k resnet50v1-5-fp32-training/quickstart/k8s/mlops/multi-node apply
```

Within the multi-node use case, a number of kustomize processing directives are enabled.

- An 'environment file' mlops.env that includes variables that will be injected as environment variables within the pod.
- The mlops user should change 'ROOT' to point to where the model package was extracted.

```
REGISTRY=docker.io
DATASET_DIR=/tf_dataset
MODEL_DIR=ROOT/resnet50v1-5-fp32-training
OUTPUT_DIR=ROOT/resnet50v1-5-fp32-training/output
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
kubectl -k resnet50v1-5-fp32-training/quickstart/k8s/mlops/single-node apply
```

Within the single-node use case, the same number of kustomize processing directives are enabled as the multi-node.

- An 'environment file' mlops.env that includes variables that will be injected as environment variables within the pod.
- The mlops user should change 'ROOT' to point to where the model package was extracted.

```
REGISTRY=docker.io
DATASET_DIR=/tf_dataset
MODEL_DIR=ROOT/resnet50v1-5-fp32-training
OUTPUT_DIR=ROOT/resnet50v1-5-fp32-training/output
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
kubectl -k resnet50v1-5-fp32-training/quickstart/k8s/mlops/multi-node delete
```

Remove the mpi-operator after running the quickstart by running the following command with
the `mpi-operator.yaml` file that you originally deployed with:

```
kubectl delete -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.2.3/deploy/v1alpha2/mpi-operator.yaml
```


!<--- 61. Advanced Options -->

See the [Advanced Options for Model Packages and Containers](ModelPackagesAdvancedOptions.md)
document for more advanced use cases.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

