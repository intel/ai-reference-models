<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the model <mode> package to get the yaml and config
files for running <mode> on a single node using Kubernetes.
```
wget <package url>
tar -xvf <package name>
```

### Execution

The Kubernetes* package for `<model name> <precision> <mode>` includes single-node and pipeline kubernetes deployments.
Within the single and pipeline deployments are common use cases that include storage and security variations that are common 
across different kubernetes installations. The directory tree within the kubernetes package is shown below, where 
single-node and pipeline directories are below the [mlops](https://en.wikipedia.org/wiki/MLOps) directory:
Common use cases are found under the single and pipeline directories:

```
quickstart
└── mlops
    ├── pipeline
    │       ├── user-allocated-pvc
    │       └── user-mounted-nfs
    └── single-node
            ├── user-allocated-pvc
            └── user-mounted-nfs
```

#### Prerequisites

Both single-node and pipeline deployments use [kustomize-v3.8.7](https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv3.8.7) to configure deployment parameters. Kustomize-v3.8.7 should be downloaded, extracted and moved to a directory within your PATH. You can verify that you've installed the correct version of kustomize by typing `kustomize version`. On MACOSX you would see:

```
{Version:kustomize/v3.8.7 GitCommit:ad092cc7a91c07fdf63a2e4b7f13fa588a39af4f BuildDate:2020-11-11T23:19:38Z GoOs:darwin GoArch:amd64}
```

#### Single-node Training

Single node training is similar to the docker use case but is run within a pod.
Training is done by submitting a pod.yaml to the k8s api-server which results in the pod creation and running
the [`launch_benchmark.py`](mlops/single-node/user-mounted-nfs/pod.yaml#L50) script within the pod's container.

In a terminal, `cd` to the single-node directory. Each use case under this directory has parameters that can be changed 
using kustomize's [cfg set](https://github.com/kubernetes-sigs/kustomize/blob/master/cmd/config/docs/commands/set.md)

##### User mounted nfs and user allocated pvc parameter values

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

Viewing the log output of the <package name> Pod is done by viewing the logs of the 
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

#### Model Training and TF Serving Pipeline

This pipeline runs the following steps using an Argo workflow:
1. Train the model on a single node using the `fp32_training_check_accuracy.sh`
   script. This script runs model training for a specified number of steps,
   exports the saved model, and compares the accuacy against the value
   specified in the `TARGET_ACCURACY` environment variable. If the model's
   accuracy does not meet the target accuracy, this step will be retried
   and continues training based on previous checkpoints in the specified
   `CHECKPOINT_DIR`. If the `TARGET_ACCURACY` environment variable has
   not been defined, then no accuracy check is done and it will continue
   on to the next step, regardless of the model's accuracy.
1. Deploy TensorFlow Serving containers with the saved model
1. Create a service that exposes the TensorFlow Serving containers as a
   NodePort

The TensorFlow Serving steps in this pipeline follows the
[TensorFlow Serving with Kubernetes instructions](https://www.tensorflow.org/tfx/serving/serving_kubernetes)
with the exception that it does not use a Google Cloud Kubernetes
cluster. Since the Kubernetes cluster being used does not have a load
balancer, the configuration is setup for NodePort, which will allow
external requests.
In a terminal, `cd` to the multi-node directory. Each use case under this directory has parameters that can be changed 
using kustomize's [cfg set](https://github.com/kubernetes-sigs/kustomize/blob/master/cmd/config/docs/commands/set.md)

##### [Mlops](https://en.wikipedia.org/wiki/MLOps)

###### User mounted nfs and user allocated pvc parameter values

|     NAME        |                 VALUE    |        DESCRIPTION          |
|-----------------|--------------------------|-----------------------------|
| DATASET_DIR     | /datasets                | input dataset directory     |
| FS_ID           | 0                        | owner id of mounted volumes |
| GROUP_ID        | 0                        | process group id            |
| GROUP_NAME      | root                     | process group name          |
| NFS_PATH        | /nfs                     | nfs path                    |
| NFS_SERVER      | 0.0.0.0                  | nfs server                  |
| PVC_NAME        | workdisk                 | model-builder | pvc name    |
| PVC_PATH        | /pvc                     | model-builder | pvc path    |
| REPLICAS        | 3                        | replica number              |
| RETRY_LIMIT     | 10                       | replica number              |
| TARGET_ACCURACY | 0.74                     | target accuracy             |
| TF_SERVING_PORT | 8501                     | tf serving port             |
| USER_ID    | 0                             | process owner id            |
| USER_NAME  | root                          | process owner name          |

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

In both use cases, the user should change the values below so the argo workflow[^3] is deployed with the user's identity.

[^3]: In order for the argo workflow to run as a non root user it must set the WorkflowExecutor to be k8sapi, otherwise the workflow will fail with "Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock". See argo issue [2239](https://github.com/argoproj/argo/issues/2239). Setting argo's WorkflowExecutor to k8sapi is described [here](https://argoproj.github.io/argo/workflow-executors/). This must be performed by devops.

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
$ kubectl get wf
$ kubectl get pods
$ kubectl logs <pod name> main
```

##### TensorFlow Serving Client

Once all the steps in the workflow have completed, the TensorFlow
Serving GRPC client can be used to run inference on the served model.

Prior to running the client script, install the following dependency in
your environment:
* tensorflow-serving-api

The client script reads a csv file (in this example we are using the
[eval.csv file](#dataset)), formats the data in for input parameter, and
then calls the served model. Accuracy and benchmarking metrics are
printed out.

Run the [run_tf_serving_client.py](run_tf_serving_client.py) script with
the `--help` flag to see the argument options:
```
$ python run_wide_deep_client.py --help
usage: <package dir>/quickstart/run_tf_serving_client.py [-h]
       [-s SERVER] -d DATA_FILE [-b BATCH_SIZE] [-n NUM_ITERATION] [-w WARM_UP_ITERATION]

optional arguments:
  -h, --help            show this help message and exit
  -s SERVER, --server SERVER
                        Server URL and port (default=localhost:8500).
  -d DATA_FILE, --data_file DATA_FILE
                        Path to csv data file
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size to use (default=1).
  -n NUM_ITERATION, --num_iteration NUM_ITERATION
                        Number of times to repeat (default=40).
  -w WARM_UP_ITERATION, --warm_up_iteration WARM_UP_ITERATION
                        Number of initial iterations to ignore in benchmarking (default=10).
```

1. Find the `INTERNAL-IP` one of the nodes in your cluster using
   `kubectl get nodes -o wide`. This IP should be used as the server URL
   in the `--server` arg.

1. Get the `NodePort` using `kubectl describe service`. This `NodePort`
   should be used as the port in the `--server` arg.

1. Run the client script with your preferred parameters. For example:
   ```
   python <package dir>/quickstart/run_tf_serving_client.py -s <Internal IP>:<Node Port> -d <path to eval.csv> --b <batch size>
   ```
   The script will call the served model using data from the csv file
   and output performance and accuracy metrics.

##### Clean up the pipeline

To clean up the model training/serving pipeline, delete the service,
deployment, and other resources using the following commands:

```
kubectl delete -f <use-case>.yaml
```
