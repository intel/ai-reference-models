<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the model training package to get the yaml and config
files for running multi-node inference on TensorFlow Serving using Kubernetes.
```
wget <package url>
tar -xvf <package name>
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
- the mpi-operator has been deployed on the cluster
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
