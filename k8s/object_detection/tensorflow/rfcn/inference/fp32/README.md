<!--- 0. Title -->
# RFCN FP32 inference

<!-- 10. Description -->

This document has instructions for running RFCN FP32 inference using
Intel® Optimizations for TensorFlow* on Kubernetes*.

<!--- 20. Download link -->
## Download link

[rfcn-fp32-inference-k8s.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/rfcn-fp32-inference-k8s.tar.gz)

<!--- 30. Datasets -->
## Dataset

The [COCO validation dataset](http://cocodataset.org) is used in these
RFCN quickstart scripts. The inference quickstart scripts use raw images,
and the accuracy quickstart scripts require the dataset to be converted
into the TF records format.
See the [COCO dataset](/datasets/coco/README.md) for instructions on
downloading and preprocessing the COCO validation dataset.


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_inference.sh`](mlops/serving/user-mounted-nfs/pod.yaml#L16) | Runs inference on a directory of raw images for 500 steps and outputs performance metrics. |
| [`fp32_accuracy.sh`](mlops/pipeline/user-mounted-nfs/serving_accuracy.yaml#L49) | Processes the TF records to run inference and check accuracy on the results. |

These quickstart scripts can be run in the following environment:
* [Kubernetes](#kubernetes)


<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the RFCN FP32 inference package.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/rfcn-fp32-inference-k8s.tar.gz
tar -xvf rfcn-fp32-inference-k8s.tar.gz
```

The Kubernetes* package for `RFCN FP32 inference` includes serving and pipeline kubernetes deployments.
Within the serving and pipeline deployments are common use cases that include storage and security
variations that are common across different kubernetes installations. The directory tree within the model package is shown below, 
where serving and pipeline directories are below the [mlops](https://en.wikipedia.org/wiki/MLOps) directory:

```
quickstart
└── mlops
    ├── pipeline
    │       ├── user-allocated-pvc
    │       └── user-mounted-nfs
    └── serving
            ├── user-allocated-pvc
            └── user-mounted-nfs
```


The `pipeline` example can be used to preprocess the coco dataset to get a
TF records file and then run an RFCN FP32 accuracy test using an
[argo workflow](https://github.com/argoproj/argo). Deployment of argo needs to be done by devops.

The `serving` example uses a pod to run inference to get performance metrics (using raw
images from the coco dataset) or test accuracy (when you already have the TF records file on NFS).

The deployments use [kustomize](https://kustomize.io/) to configure
parameters. The parameters can be set by running kustomize commands
prior to deploying the job to kubernetes.

#### Prerequisites

The rfcn-fp32-inference-k8s.tar.gz package uses [kustomize-v3.8.7](https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv3.8.7) to configure parameters within the deployment.yaml. Kustomize-v3.8.7 should be downloaded, extracted and moved to a directory within your PATH. You can verify that you've installed the correct version of kustomize by typing `kustomize version`. On MACOSX you would see:

```
{Version:kustomize/v3.8.7 GitCommit:ad092cc7a91c07fdf63a2e4b7f13fa588a39af4f BuildDate:2020-11-11T23:19:38Z GoOs:darwin GoArch:amd64}
```

#### Serving Inference

Inference is run by submitting a pod yaml file to the k8s api-server,
which results in the pod creation and then the specified
[quickstart script](#quick-start-scripts) is run in the pod's container.

Make sure you are inside the serving directory:

```
cd rfcn-fp32-inference-k8s/quickstart/mlops/serving
```

The parameters that can be changed within the serving deployment are shown in the table below:

|     NAME     |             VALUE              |         DESCRIPTION         |
|--------------|--------------------------------|-----------------------------|
| DATASET_DIR  | /datasets                      | input dataset directory     |
| FS_ID        | 0                              | owner id of mounted volumes |
| GROUP_ID     | 0                              | process group id            |
| GROUP_NAME   | root                           | process group name          |
| NFS_PATH     | /nfs                           | nfs path                    |
| NFS_SERVER   | 0.0.0.0                        | nfs server                  |
| PVC_NAME     | workdisk                       | pvc name                    |
| PVC_PATH     | /pvc                           | pvc path                    |
| OUTPUT_DIR   | output                         | output dir basename         |
| USER_ID      | 0                              | process owner id            |
| USER_NAME    | root                           | process owner name          |

> Note that when running inference, the `DATASET_DIR` should point to the
> directory of raw coco images (val2017) and when running accuracy testing,
> the `DATASET_DIR` should point to the TF records directory.

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

##### Serving inference output

Viewing the log output of the rfcn-fp32-inference-k8s.tar.gz is done by viewing the logs of the
deployed pod. This pod is found by filtering the list of pods for the name 'inference'

```
kubectl get pods -oname|grep inference|cut -c5-
```

This can be combined with the kubectl logs subcommand to tail the output of the inference job

```
kubectl logs -f $(kubectl get pods -oname|grep inference|cut -c5-)
```

##### Serving inference cleanup

Removing the pod and related resources is done by running:

```
kubectl delete -f <use-case>.yaml
```

#### Pipeline

The pipeline job uses an [Argo workflow](https://github.com/argoproj/argo)
to first convert the raw coco images to the TF records format and then
runs RFCN FP32 inference with an accuracy test using the TF records file.

The [COCO validation 2017 dataset and annotations](https://cocodataset.org/#download)
need to be downloaded to a directory on nfs. These will be used to create
the TF records file.

Make sure you are inside the pipeline directory:

```
cd rfcn-fp32-inference-k8s/quickstart/mlops/pipeline
```

The parameters that can be changed within the pipeline are shown in the table below:

|       NAME        |             VALUE              |          DESCRIPTION           |
|-------------------|--------------------------------|--------------------------------|
| DATASET_DIR       | /datasets                      | input dataset directory        |
| FS_ID             | 0                              | owner id of mounted volumes    |
| GROUP_ID          | 0                              | process group id               |
| GROUP_NAME        | root                           | process group name             |
| NFS_PATH          | /nfs                           | nfs path                       |
| NFS_SERVER        | 0.0.0.0                        | nfs server                     |
| PVC_NAME          | workdisk                       | pvc name                       |
| PVC_PATH          | /pvc                           | pvc path                       |
| OUTPUT_DIR        | output                         | output dir basename            |
| USER_ID           | 0                              | process owner id               |
| USER_NAME         | root                           | process owner name             |

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

In both use cases, the user should change the values below so the pod is deployed with the user's identity[^3].

[^3]: In order for the argo workflow to run as a non root user it must set the WorkflowExecutor to be k8sapi, otherwise the workflow will fail with "Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock". See argo issues [2239](https://github.com/argoproj/argo/issues/2239),[4186](https://github.com/argoproj/argo/issues/4186). Setting argo's WorkflowExecutor to k8sapi is described [here](https://argoproj.github.io/argo/workflow-executors/). This must be performed by devops.

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

Once the job has been submitted, the status and logs can be viewed using
the Argo user inferface or from the command line using kubectl or argo.
The commands below describe how to use kubectl to see the workflow, pods,
and log files:
```
kubectl get wf
kubectl get pods
kubectl logs <pod name> main
```

##### Cleanup

Remove the workflow and related resources using the following command:

```
kubectl delete -f object_detection.yaml
```

### Advanced Options

See the [Advanced Options for Model Packages and Containers](/quickstart/common/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.

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

