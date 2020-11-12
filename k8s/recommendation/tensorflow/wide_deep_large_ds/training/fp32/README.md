<!--- 0. Title -->
# Wide and Deep Large Dataset FP32 training

<!-- 10. Description -->

This document has instructions for running [Wide and Deep](https://arxiv.org/pdf/1606.07792.pdf) FP32 training using
Intel® Optimizations for TensorFlow* on Kubernetes*.



<!--- 20. Download link -->
## Download link

[wide-deep-large-ds-fp32-training-k8s.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_1_0/wide-deep-large-ds-fp32-training-k8s.tar.gz)

<!--- 30. Datasets -->
## Dataset

The large [Kaggle Display Advertising Challenge Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
will be used for training Wide and Deep. The
[data](https://www.kaggle.com/c/criteo-display-ad-challenge/data) is from
[Criteo](https://www.criteo.com) and has a field indicating if an ad was
clicked (1) or not (0), along with integer and categorical features.

Download large Kaggle Display Advertising Challenge Dataset from
[Criteo Labs](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
* Download the large version of train dataset from: https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
* Download the large version of evaluation dataset from: https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv

The directory where you've downloaded the `train.csv` and `eval.csv`
files should be used as the `DATASET_DIR` when running [quickstart scripts](#quick-start-scripts).


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_training_check_accuracy.sh`](mlops/pipeline/fp32_training_check_accuracy.sh) | Trains the model for a specified number of steps (default is 500) and then compare the accuracy against the specified target accuracy. If the accuracy is not met, then script exits with error code 1. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |
| [`fp32_training.sh`](mlops/single-node/fp32_training.sh) | Trains the model for 10 epochs. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |
| [`run_tf_serving_client.py`](run_tf_serving_client.py) | Runs gRPC client for multi-node batch and online inference. |

These quickstart scripts can be run in the following environment:
* [Kubernetes](#kubernetes)

<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the model training package to get the yaml and config
files for running training on a single node using Kubernetes.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_1_0/wide-deep-large-ds-fp32-training-k8s.tar.gz
tar -xvf wide-deep-large-ds-fp32-training-k8s.tar.gz
```

### Execution

The Kubernetes* package for `Wide and Deep Large Dataset FP32 training` includes single-node and pipeline kubernetes deployments.
The directory tree within the kubernetes package is shown below, where single-node and pipeline directories are below the
[mlops](https://en.wikipedia.org/wiki/MLOps) directory:

```
quickstart
└── mlops
    ├── pipeline
    └── single-node
```

#### Prerequisites

Both single-node and pipeline deployments use [kustomize-v3.8.4](https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv3.8.4) to configure deployment parameters. This archive should be downloaded, extracted and the kustomize command should be moved to a directory within your PATH. You can verify the correct version of kustomize has been installed by typing `kustomize version`. On MACOSX you would see

```
{Version:kustomize/v3.8.4 GitCommit:8285af8cf11c0b202be533e02b88e114ad61c1a9 BuildDate:2020-09-19T15:39:21Z GoOs:darwin GoArch:amd64}
```


The kustomization files that the kustomize command references are located withing the following directories:

```
wide-deep-large-ds-fp32-training-k8s.tar.gz/quickstart/mlops/pipeline/kustomization.yaml
wide-deep-large-ds-fp32-training-k8s.tar.gz/quickstart/mlops/single-node/kustomization.yaml
```

#### Single-node Training

Single node training is similar to the docker use case but the command is run within a pod.
Training is done by submitting a pod.yaml to the k8s api-server which results in the pod creation and running
the quick start script command within the pod's container.

Make sure you are inside the single-node directory:

```
cd wide-deep-large-ds-fp32-training-k8s/quickstart/mlops/single-node
```

The parameters that can be changed within the pod are shown in the table[^1] below:

|     NAME     |                    VALUE                    |    SET BY     |        DESCRIPTION        | COUNT | REQUIRED |
|--------------|---------------------------------------------|---------------|---------------------------|-------|----------|
| DATASET_DIR  | /datasets                                   | model-builder | input dataset directory   | 3     | Yes      |
| FS_ID        | 0                                           | model-builder | filesystem id             | 1     | Yes      |
| GROUP_ID     | 0                                           | model-builder | process group id          | 2     | Yes      |
| GROUP_NAME   | root                                        | model-builder | process group name        | 1     | Yes      |
| IMAGE_SUFFIX |                                             | model-builder | appended to image name    | 1     | No       |
| MODEL_DIR    | /workspace/wide-deep-large-ds-fp32-training | model-builder | container model directory | 3     | No       |
| MODEL_NAME   | wide-deep-large-ds-fp32-training            | model-builder | name use-case             | 5     | No       |
| MODEL_SCRIPT | fp32_training.sh                            | model-builder | model script name         | 5     | No       |
| NFS_PATH     | /nfs                                        | model-builder | nfs path                  | 3     | Yes      |
| NFS_SERVER   | 0.0.0.0                                     | model-builder | nfs server                | 1     | Yes      |
| OUTPUT_DIR   | output                                      | model-builder | output dir base name      | 1     | Yes      |
| REGISTRY     | docker.io                                   | model-builder | image location            | 1     | No       |
| USER_ID      | 0                                           | model-builder | process owner id          | 2     | Yes      |
| USER_NAME    | root                                        | model-builder | process owner name        | 2     | Yes      |


[^1]: The single-node parameters table is generated by `kustomize cfg list-setters . --markdown`. See [list-setters](https://github.com/kubernetes-sigs/kustomize/blob/master/cmd/config/docs/commands/list-setters.md) for explanations of each column.

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

This will tell kubernetes to deploy the resources within the specified namespace. Note: this namespace should be created prior to deployment.
Once the user has changed parameter values they can then deploy the single-node job by running:

```
kustomize build > single-node.yaml
kubectl apply -f single-node.yaml
```

##### Single-node training output

The script will write a log file, checkpoints, and the saved model to
the `OUTPUT_DIR`.

Viewing the log output of the single-node job is done by viewing the logs of the
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
kubectl delete -f single-node.yaml
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

Make sure you are inside the pipeline directory:

```
cd wide-deep-large-ds-fp32-training-k8s/quickstart/mlops/pipeline
```

The parameters that can be changed within the pipeline are shown in the table[^2] below:

|      NAME       |                    VALUE                    |    SET BY     |         DESCRIPTION         | COUNT | REQUIRED |
|-----------------|---------------------------------------------|---------------|-----------------------------|-------|----------|
| CHECKPOINT_DIR  | checkpoints                                 | model-builder | checkpoint dir basename     | 1     | Yes      |
| DATASET_DIR     | /datasets                                   | model-builder | input dataset directory     | 3     | Yes      |
| FS_ID           | 0                                           | model-builder | owner id of mounted volumes | 2     | Yes      |
| GROUP_ID        | 0                                           | model-builder | process group id            | 3     | Yes      |
| GROUP_NAME      | root                                        | model-builder | process group name          | 1     | Yes      |
| IMAGE_SUFFIX    |                                             | model-builder | appended to image name      | 1     | No       |
| MODEL_DIR       | /workspace/wide-deep-large-ds-fp32-training | model-builder | container model directory   | 3     | No       |
| MODEL_NAME      | wide-deep-large-ds-fp32-training            | model-builder | name use-case               | 9     | No       |
| MODEL_SCRIPT    | fp32_training_check_accuracy.sh             | model-builder | model script name           | 5     | No       |
| NFS_PATH        | /nfs                                        | model-builder | nfs path                    | 6     | Yes      |
| NFS_SERVER      | 0.0.0.0                                     | model-builder | nfs server                  | 2     | Yes      |
| OUTPUT_DIR      | output                                      | model-builder | output dir basename         | 2     | Yes      |
| REGISTRY        | docker.io                                   | model-builder | image location              | 1     | No       |
| REPLICAS        | 3                                           | model-builder | replica number              | 1     | No       |
| RETRY_LIMIT     | 10                                          | model-builder | replica number              | 1     | No       |
| TARGET_ACCURACY | 0.75                                        | model-builder | target accuracy             | 1     | Yes      |
| TF_SERVING_PORT | 8501                                        | model-builder | tf serving port             | 1     | Yes      |
| USER_ID         | 0                                           | model-builder | process owner id            | 3     | Yes      |
| USER_NAME       | root                                        | model-builder | process owner name          | 4     | Yes      |

[^2]: The pipeline parameters table is generated by `kustomize cfg list-setters . --markdown`. See [list-setters](https://github.com/kubernetes-sigs/kustomize/blob/master/cmd/config/docs/commands/list-setters.md) for explanations of each column.

For example to change the NFS_SERVER IP address to 10.35.215.25 the user would run:

```
kustomize cfg set . NFS_SERVER 10.35.215.25
```

The required column that contains a 'Yes' indicates which values should be changed by the user.
The 'No' values indicate that the default values are fine. Note that the mlops user should run the
argo workflow[^3] with their own uid/gid permissions by using kustomize to change the securityContext in the single_node_accuracy.yaml file.

This is done by running the following:

```
kustomize cfg set . FS_ID <Group ID>
kustomize cfg set . GROUP_ID <Group ID>
kustomize cfg set . GROUP_NAME <Group Name>
kustomize cfg set . USER_ID <User ID>
kustomize cfg set . USER_NAME <User Name>
```

[^3]: In order for the argo workflow to run as a non root user it must set the WorkflowExecutor to be k8sapi, otherwise the workflow will fail with "Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock". See argo issue [2239](https://github.com/argoproj/argo/issues/2239). Setting argo's WorkflowExecutor to k8sapi is described [here](https://argoproj.github.io/argo/workflow-executors/). This must be performed by devops.

Finally, the namespace can be changed by the user from the default namespace by running the kustomize command:

```
kustomize edit set namespace $USER
```

This will tell kubernetes to deploy the resources within the specified namespace. Note: this namespace should be created prior to deployment.
Once the user has changed parameter values they can then deploy the workflow by running:

```
kustomize build > pipeline.yaml
kubectl apply -f pipeline.yaml
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
usage: wide-deep-large-ds-fp32-training-k8s/quickstart/run_tf_serving_client.py [-h]
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
   python wide-deep-large-ds-fp32-training-k8s/quickstart/run_tf_serving_client.py -s <Internal IP>:<Node Port> -d <path to eval.csv> --b <batch size>
   ```
   The script will call the served model using data from the csv file
   and output performance and accuracy metrics.

##### Clean up the pipeline

To clean up the model training/serving pipeline, delete the service,
deployment, and other resources using the following commands:
```
kubectl delete -f pipeline.yaml
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

