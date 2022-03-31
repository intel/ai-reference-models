<!--- 0. Title -->
# ResNet50 v1.5 FP32 inference

<!-- 10. Description -->

This document has instructions for running ResNet50 v1.5 FP32 inference using
Intel® Optimizations for TensorFlow* on Kubernetes*.



<!--- 20. Download link -->
## Download link

[resnet50v1-5-fp32-inference-k8s.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50v1-5-fp32-inference-k8s.tar.gz)

<!--- 30. Datasets -->
## Dataset

OPTIONAL: The ImageNet dataset can be used in these ResNet50 v1.5 Kubernetes examples.
You can download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`run_tf_serving_client.py`](run_tf_serving_client.py) | Runs gRPC client for multi-node batch and online inference. |
| [`multi_client.sh`](multi_client.sh) | Runs multiple parallel gRPC clients for multi-node batch and online inference. |

These quickstart scripts can be run in this environment:
* [Kubernetes](#kubernetes)


<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the ResNet50 v1.5 FP32 inference package.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50v1-5-fp32-inference-k8s.tar.gz
tar -xvf resnet50v1-5-fp32-inference-k8s.tar.gz
```

### Execution

The Kubernetes* package for `ResNet50 v1.5 FP32 inference` includes a kubernetes serving deployment.
The directory tree within the model package is shown below, where the serving directory is below the
[mlops](https://en.wikipedia.org/wiki/MLOps) directory:

```
quickstart
└── mlops
    └── serving
```

#### Prerequisites

The resnet50v1-5-fp32-inference-k8s.tar.gz package uses [kustomize-v3.8.7](https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv3.8.7) to configure parameters within the deployment.yaml. Kustomize-v3.8.7 should be downloaded, extracted and moved to a directory within your PATH. You can verify that you've installed the correct version of kustomize by typing `kustomize version`. On MACOSX you would see:

```
{Version:kustomize/v3.8.7 GitCommit:ad092cc7a91c07fdf63a2e4b7f13fa588a39af4f BuildDate:2020-11-11T23:19:38Z GoOs:darwin GoArch:amd64}
```

#### TensorFlow Serving

The multi-node serving use case makes the following assumptions:
- the model directory is a shared volume mounted cluster wide
- a saved model has been generated and saved under the model directory volume

TensorFlow Serving is run by submitting a deployment and service yaml file to the k8s api-server,
which results in the creation of pod replicas, each serving the model on a different node of the cluster.

The steps follow the
[TensorFlow Serving with Kubernetes instructions](https://www.tensorflow.org/tfx/serving/serving_kubernetes)
with the exception that it does not use a Google Cloud Kubernetes
cluster. Since the Kubernetes cluster being used does not have a load
balancer, the configuration is setup for NodePort, which will allow
external requests.

Make sure you are inside the serving directory:

```
cd resnet50v1-5-fp32-inference-k8s/quickstart/mlops/serving
```

The parameters that should be changed within the serving resources are shown in the table below:

|            NAME             |                  VALUE                   |         DESCRIPTION         |
|-----------------------------|------------------------------------------|-----------------------------|
| FS_ID                       | 0                                        | owner id of mounted volumes |
| GROUP_ID                    | 0                                        | process group id            |
| GROUP_NAME                  | root                                     | process group name          |
| MODEL_BASE_PATH             | /models                                  | mounted model directory     |
| MODEL_DIR                   | /models                                  | host model base directory   |
| MODEL_NAME                  | resnet50v1_5                             | model name                  |
| MODEL_PORT                  | 8500                                     | model container port        |
| MODEL_SERVICE_PORT          | 8501                                     | model service port          |
| REPLICAS                    | 3                                        | number of replicas          |
| USER_ID                     | 0                                        | process owner id            |
| USER_NAME                   | root                                     | process owner name          |


For example to change the MODEL_SERVICE_PORT from 8500 to 9500

```
kustomize cfg set . MODEL_SERVICE_PORT 9500
```

The user should change the values below so the pod is deployed with the user's identity, rather than running as root.

```
kustomize cfg set . FS_ID <Group ID> -R
kustomize cfg set . GROUP_ID <Group ID> -R
kustomize cfg set . GROUP_NAME <Group Name> -R
kustomize cfg set . USER_ID <User ID> -R
kustomize cfg set . USER_NAME <User Name> -R
```

The user should change the values below so the SavedModel is mounted into the pod from the host machine.

```
kustomize cfg set . MODEL_BASE_PATH <Full path in the pod where the model directory will be mounted> -R
kustomize cfg set . MODEL_DIR <Full path on the host to the directory containing the model directory> -R
kustomize cfg set . MODEL_NAME <Model name - must be the name of the directory containing the SavedModel> -R
```

The user should change the default namespace of all the resources by running the kustomize command:

```
kustomize edit set namespace <User's namespace>
```

This will place all resources within the specified namespace. Note: this namespace should be created prior to deployment.

The user can also change their default kubectl context by running

```
kubectl config set-context --current --namespace=<User's namespace>
```

Once the user has changed parameter values they can then deploy the resnet50v1-5-fp32-inference-k8s.tar.gz by running:

```
kustomize build  . | kubectl apply -f -
```

##### TensorFlow Serving Client

Once all the pods are running, the TensorFlow
Serving gRPC client can be used to run inference on the served model.

Prior to running the client script, install the following dependency in
your environment:
* tensorflow-serving-api

The client script accepts an optional argument `--data_dir` pointing to a directory of images in TF records format. 
If the argument is not present, dummy data will be used instead. Then the script formats the data as a gRPC request object and
calls the served model API. Benchmarking metrics are printed out.

Run the [run_tf_serving_client.py](run_tf_serving_client.py) script with
the `--help` flag to see the argument options:
```
$ python resnet50v1-5-fp32-inference-k8s.tar.gz/quickstart/run_tf_serving_client.py --help
Send TF records or simulated image data to tensorflow_model_server loaded with ResNet50v1_5 model.

flags:

run_tf_serving_client.py:
  --batch_size: Batch size to use
    (default: '1')
    (an integer)
  --data_dir: Path to images in TF records format
    (default: '')
  --server: PredictionService host:port
    (default: 'localhost:8500')
```

1. Find the `INTERNAL-IP` of one of the nodes in your cluster using
   `kubectl get nodes -o wide`. This IP should be used as the server URL
   in the `--server` arg.

1. Get the `NodePort` using `kubectl describe service`. This `NodePort`
   should be used as the port in the `--server` arg.

1. Run the client script with your preferred parameters. For example:
   ```
   python resnet50v1-5-fp32-inference-k8s.tar.gz/quickstart/run_tf_serving_client.py --server <Internal IP>:<Node Port> --data_dir <path to TF records> --batch_size <batch size>
   ```
   The script will call the served model using data from the `data_dir` path or simulated data
   and output performance metrics.
   
1. If you want to send multiple parallel calls to the server, you can use the `multi_client.sh` script.
   Its options are:
   ```
   --model: Model to be inferenced
   --batch_size: Number of samples per batch (default 1)
   --servers: "server1:port;server2:port" (default "localhost:8500")
   --clients: Number of clients per server (default 1)
   ```
   
   You must be in the quickstart folder to run it, and the script does not currently support a `data_dir` argument (only runs with simulated data).
   Example:
   
   ```
   cd resnet50v1-5-fp32-inference-k8s.tar.gz/quickstart
   bash multi_client.sh --model=resnet50v1_5 --servers="<Internal IP>:<Node Port>" --clients=5
   cd ../../
   ```

##### Clean up the pipeline

To clean up the served model, delete the service,
deployment, and other resources using the following commands:
```
kustomize build  . | kubectl delete -f -
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

