<!--- 0. Title -->
# ResNet50 v1.5 FP32 inference

<!-- 10. Description -->

This document has instructions for running ResNet50 v1.5 FP32 inference using
Intel-optimized TensorFlow.

Note that the ImageNet dataset is used in these ResNet50 v1.5 examples.
Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.


<!--- 20. Download link -->
## Download link

[resnet50v1-5-fp32-inference.tar.gz](https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/resnet50v1-5-fp32-inference.tar.gz)

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](fp32_online_inference.sh) | Runs online inference (batch_size=1). |
| [`fp32_batch_inference.sh`](fp32_batch_inference.sh) | Runs batch inference (batch_size=128). |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | Measures the model accuracy (batch_size=100). |
| [`run_tf_serving_client.py`](run_tf_serving_client.py) | Runs gRPC client for multi-node batch and online inference. |
| [`multi_client.sh`](multi_client.sh) | Runs multiple parallel gRPC clients for multi-node batch and online inference. |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
* [Kubernetes](#kubernetes)


<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow==2.1.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package and then run a [quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/resnet50v1-5-fp32-inference.tar.gz
tar -xzf resnet50v1-5-fp32-inference.tar.gz
cd resnet50v1-5-fp32-inference

quickstart/<script name>.sh
```


<!-- 60. Docker -->
## Docker

The model container `amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-image-recognition-resnet50v1-5-fp32-inference` includes the scripts
and libraries needed to run ResNet50 v1.5 FP32 inference. To run one of the model
inference quickstart scripts using this container, you'll need to provide volume mounts for
the ImageNet dataset and an output directory where checkpoint files will be written.

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-image-recognition-resnet50v1-5-fp32-inference \
  /bin/bash quickstart/<script name>.sh
```


<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the model training package to get the yaml and config
files for running multi-node inference on TensorFlow Serving using Kubernetes.
```
wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/resnet50v1-5-fp32-inference.tar.gz
tar -xvf resnet50v1-5-fp32-inference.tar.gz
```

### Execution

The model package includes a deployment that does 'mlops' (machine learning
operations) on kubernetes.
The relevant directory tree within the model package is shown below:
```
quickstart
├── common
│   └── tensorflow
│       └── k8s
│           └── mlops
│               └── serving
└── k8s
    └── mlops
        └── serving
```

The multi-node serving use case makes the following assumptions:
- the nfs share is available cluster wide
- a saved model has been generated and saved to a nfs shared volume

The parameters are configured by editing kustomize related files described in the sections below.
The deployment uses [kustomize](https://kustomize.io/) to configure
parameters. The parameters can be customized by editing kustomize
related files prior to deploying the serving job, which
is described in the [next section](#tensorflow-serving).

#### TensorFlow Serving

TensorFlow Serving is run by submitting a deployment and service yaml file to the k8s api-server,
which results in the creation of pod replicas, each serving the model on a different node of the cluster.

The steps follow the
[TensorFlow Serving with Kubernetes instructions](https://www.tensorflow.org/tfx/serving/serving_kubernetes)
with the exception that it does not use a Google Cloud Kubernetes
cluster. Since the Kubernetes cluster being used does not have a load
balancer, the configuration is setup for NodePort, which will allow
external requests.

Prior to running the job, edit the kustomize variables in the mlops.env
file. 

The mlops.env file for TensorFlow Serving jobs is located at:
`resnet50v1-5-fp32-inference/quickstart/k8s/mlops/serving/mlops.env`.
Key parameters to edit are:
```
MODEL_NAME=<Name of the model which is also the parent directory of the saved model, i.e. resnet50v1_5>
MODEL_DIR=<Host path to mount inside the container which contains the saved model>
MODEL_BASE_PATH=<Host path containing a directory named MODEL_NAME which contains the saved model>
MODEL_PORT=<Container port to use for TF serving>
MODEL_SERVICE_PORT=<Container port to use for NodePort service>
REPLICAS=<Number of TF serving replicas to deploy>
```

Once you have edited the `mlops.env` file with your parameters,
deploy the training job using the command below. This command will
deploy resources to your default namespace. To use a different
namespace, specify `-n <namespace>` as part of your command.
```
kubectl -k resnet50v1-5-fp32-inference/quickstart/k8s/mlops/serving apply
```

Depending on what version of kustomize is being used, you may get an
error reporting that a string was received instead of a integer. If this
is the case, the following command can be used to remove quotes that
are causing the issue:
```
kubectl kustomize resnet50v1-5-fp32-inference/quickstart/k8s/mlops/serving | sed 's/port:.*"\([0-9]*\)"/port: \1/g' | sed 's/targetPort:.*"\([0-9]*\)"/targetPort: \1/g' | sed 's/replicas:.*"\([0-9]*\)"/replicas: \1/g' | sed 's/containerPort:.*"\([0-9]*\)"/containerPort: \1/g' | kubectl apply -f -
```

Once the kubernetes workflow has been submitted, the status can be
checked using (you will need to substitute a pod's name in the second line):
```
kubectl get pods
kubectl logs -f <pod name>
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
$ python resnet50v1-5-fp32-inference/quickstart/run_tf_serving_client.py --help
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
   python resnet50v1-5-fp32-inference/quickstart/run_tf_serving_client.py --server <Internal IP>:<Node Port> --data_dir <path to TF records> --batch_size <batch size>
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
   cd resnet50v1-5-fp32-inference/quickstart
   bash multi_client.sh --model=resnet50v1_5 --servers="<Internal IP>:<Node Port>" --clients=5
   cd ../../
   ```

##### Clean up the pipeline

To clean up the served model, delete the service,
deployment, and other resources using the following commands:
```
kubectl -k resnet50v1-5-fp32-inference/quickstart/k8s/mlops/serving delete
```

<!-- 61. Advanced Options -->
### Advanced Options

See the [Advanced Options for Model Packages and Containers](/quickstart/common/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

