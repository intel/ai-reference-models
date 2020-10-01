<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the model <mode> package to get the yaml and config
files for running <mode> on a single node using Kubernetes.
```
wget <package url>
tar -xvf <package name>
```

### Execution

The model package includes a deployment that does 'mlops' (machine learning
operations) on kubernetes.
The directory tree within the model package is shown below:
```
quickstart
├── common
│   └── tensorflow
│       └── k8s
│           └── mlops
│               ├── base
│               └── single-node
└── k8s
    └── mlops
        ├── pipeline
        └── single-node
```

The deployments uses [kustomize](https://kustomize.io/) to configure
parameters. The parameters can be customized by editing kustomize
related files prior to deploying the single node or pipeline job, which
is described in the [next section](#single-node-training).

#### Single-node Training

Training is run by submitting a pod yaml file to the k8s api-server,
which results in the pod creation and then the specified
[quickstart script](#quick-start-scripts) is run in the pod's container.

Prior to running the job, edit the kustomize varaibles in the mlops.env
file. The mlops.env file for single node jobs is located at:
`<package dir>/quickstart/k8s/mlops/single-node/mlops.env`.
Key parameters to edit are:
```
DATASET_DIR=<path to the dataset directory>
MODEL_SCRIPT=<fp32_training.sh or another quickstart script>
NFS_MOUNT_PATH=<Path where the NFS directory will be mounted in the container>
NFS_PATH=<NFS path>
NFS_SERVER=<IP address for your NFS Server>
OUTPUT_DIR=<Directory where logs, checkpoints, and the saved model will be written>
USER_ID=<Your user ID>
GROUP_ID=<Your group ID>
```

Once you have edited the `mlops.env` file with your parameters,
deploy the training job using the following command:
```
kubectl -k <package dir>/quickstart/k8s/mlops/single-node apply
```

Depending on what version of kustomize is being used, you may get an
error reporting that a string was received instead of a integer. If this
is the case, the following command can be used to remove quotes that
are causing the issue:
```
kubectl kustomize <package dir>/quickstart/k8s/mlops/single-node | sed 's/runAsUser:.*"\([0-9]*\)"/runAsUser: \1/g' | sed 's/runAsGroup:.*"\([0-9]*\)"/runAsGroup: \1/g' | sed 's/fsGroup:.*"\([0-9]*\)"/fsGroup: \1/g' | kubectl apply -f -
```

Once the kubernetes job has been submitted, the pod status can be
checked using `kubectl get pods` and the logs can be viewed using
`kubectl logs -f wide-deep-large-ds-fp32-training`.

The script will write a log file, checkpoints, and the saved model to
the `OUTPUT_DIR`.

##### Clean up single node training

Clean up the model training job (delete the pod and other resources) using the following command:
```
kubectl -k <package dir>/quickstart/k8s/mlops/single-node delete
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

Prior to running the job, edit the kustomize varaibles in the mlops.env
file. The mlops.env file for single node jobs is located at:
`<package dir>/quickstart/k8s/mlops/pipeline/mlops.env`.
Key parameters to edit are:
```
DATASET_DIR=<path to the dataset directory>
MODEL_SCRIPT=fp32_training_check_accuracy.sh
NFS_MOUNT_PATH=<Path where the NFS directory will be mounted in the container>
NFS_PATH=<NFS path>
NFS_SERVER=<IP address for your NFS Server>
USER_ID=<Your user ID>
GROUP_ID=<Your group ID>
OUTPUT_DIR=<Directory where logs and the saved model will be written>
CHECKPOINT_DIR=<Directory where checkpoint files will be read and written>
TARGET_ACCURACY=<A decimal value between 0 and 1 (for example: .75)
STEPS=<The number of training steps>
RETRY_LIMIT=<The number of times to retry training>
REPLICAS=<Number of TF serving replicas to deploy>
TF_SERVING_PORT=<Container port to use for TF serving>
```
> Note that the `OUTPUT_DIR` and `CHECKPOINT_DIR` paths should be on
> your NFS mount path, so that they are accessible by all of the nodes.

Once you have edited the `mlops.env` file with your parameters,
deploy the training job using the command below. This command will
deploy resources to your default namespace. To use a different
namespace, specify `-n <namespace>` as part of your command.
```
kubectl -k <package dir>/quickstart/k8s/mlops/pipeline apply
```

Depending on what version of kustomize is being used, you may get an
error reporting that a string was received instead of a integer. If this
is the case, the following command can be used to remove quotes that
are causing the issue:
```
kubectl kustomize <package dir>/quickstart/k8s/mlops/pipeline | sed 's/runAsUser:.*"\([0-9]*\)"/runAsUser: \1/g' | sed 's/runAsGroup:.*"\([0-9]*\)"/runAsGroup: \1/g' | sed 's/fsGroup:.*"\([0-9]*\)"/fsGroup: \1/g' | sed 's/replicas:.*"\([0-9]*\)"/replicas: \1/g' | sed 's/containerPort:.*"\([0-9]*\)"/containerPort: \1/g' | sed 's/limit:.*"\([0-9]*\)"/limit: \1/g' | kubectl apply -f -
```

Once the kubernetes workflow has been submitted, the status can be
checked using `kubectl get wf` and `kubectl get pods`. The pod logs can
be viewed using `kubectl logs -f <pod name>`.

##### TensorFlow Serving Client

Once all the steps in the workflow have completed, the TensorFlow
Serving GRPC client can be used to run inference on the served model.

Prior to running the client script, install the following dependency in
your enviornment:
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
kubectl -k <package dir>/quickstart/k8s/mlops/pipeline delete
```
