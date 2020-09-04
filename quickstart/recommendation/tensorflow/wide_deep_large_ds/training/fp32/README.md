<!--- 0. Title -->
# Wide and Deep Large Dataset FP32 training

<!-- 10. Description -->

This document has instructions for training [Wide and Deep](https://arxiv.org/pdf/1606.07792.pdf)
using a large dataset using Intel-optimized TensorFlow.


<!--- 20. Download link -->
## Download link

[wide-deep-large-ds-fp32-training.tar.gz](https://ubit-artifactory-or.intel.com/artifactory/cicd-or-local/model-zoo/wide-deep-large-ds-fp32-training.tar.gz)

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
| [`fp32_training_500_steps.sh`](fp32_training_500_steps.sh) | Limits training to 500 steps for a shorter training run. |
| [`fp32_training.sh`](fp32_training.sh) | Trains the model for 10 epochs. |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
* [Kubernetes](#kubernetes)

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow](https://pypi.org/project/intel-tensorflow/)

Download and untar the model package and then run a
[quickstart script](#quick-start-scripts) with enviornment variables
that point to the [dataset](#dataset) and an output directory where
log files, checkpoint files, and the saved model will be written.

```
DATASET_DIR=<path to the dataset directory>
OUTPUT_DIR=<directory where the logs, checkpoints, and the saved model will be written>

wget https://ubit-artifactory-or.intel.com/artifactory/cicd-or-local/model-zoo/wide-deep-large-ds-fp32-training.tar.gz
tar -xvf wide-deep-large-ds-fp32-training.tar.gz
cd wide-deep-large-ds-fp32-training

quickstart/<script name>.sh
```

The script will write a log file, checkpoints, and the saved model to
the `OUTPUT_DIR`.

<!-- 60. Docker -->
## Docker

The model container used in the example below includes the scripts and
libraries needed to run Wide and Deep Large Dataset FP32 training. To run one of the
model quickstart scripts using this container, you'll need to provide
volume mounts for the [dataset](#dataset) and an output directory where
logs, checkpoints, and the saved model will be written.
```
DATASET_DIR=<path to the dataset directory>
OUTPUT_DIR=<directory where the logs, checkpoints, and the saved model will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-recommendation-wide-deep-large-ds-fp32-training \
  /bin/bash quickstart/<script name>.sh
```

The script will write a log file, checkpoints, and the saved model to
the `OUTPUT_DIR`.

<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the model training package to get the yaml and config
files for running training on a single node using Kubernetes.
```
wget https://ubit-artifactory-or.intel.com/artifactory/cicd-or-local/model-zoo/wide-deep-large-ds-fp32-training.tar.gz
tar -xvf wide-deep-large-ds-fp32-training.tar.gz
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
`wide-deep-large-ds-fp32-training/quickstart/k8s/mlops/single-node/mlops.env`.
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
kubectl -k wide-deep-large-ds-fp32-training/quickstart/k8s/mlops/single-node apply
```

Depending on what version of kustomize is being used, you may get an
error reporting that a string was received instead of a integer. If this
is the case, the following command can be used to remove quotes that
are causing the issue:
```
kubectl kustomize wide-deep-large-ds-fp32-training/quickstart/k8s/mlops/single-node | sed 's/runAsUser:.*"\([0-9]*\)"/runAsUser: \1/g' | sed 's/runAsGroup:.*"\([0-9]*\)"/runAsGroup: \1/g' | sed 's/fsGroup:.*"\([0-9]*\)"/fsGroup: \1/g' | kubectl apply -f -
```

Once the kubernetes job has been submitted, the pod status can be
checked using `kubectl get pods` and the logs can be viewed using
`kubectl logs -f wide-deep-large-ds-fp32-training`.

The script will write a log file, checkpoints, and the saved model to
the `OUTPUT_DIR`.

##### Clean up single node training

Clean up the model training job (delete the pod and other resources) using the following command:
```
kubectl -k wide-deep-large-ds-fp32-training/quickstart/k8s/mlops/single-node delete
```

#### Model Training and TF Serving Pipeline

This pipeline runs the following steps using an Argo workflow:
1. Model training on a single node, then export a saved model
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
`wide-deep-large-ds-fp32-training/quickstart/k8s/mlops/pipeline/mlops.env`.
Key parameters to edit are:
```
DATASET_DIR=<path to the dataset directory>
MODEL_SCRIPT=<fp32_training.sh or another quickstart script>
NFS_MOUNT_PATH=<Path where the NFS directory will be mounted in the container>
NFS_PATH=<NFS path>
NFS_SERVER=<IP address for your NFS Server>
OUTPUT_DIR=<Directory where logs, checkpoints, and the saved model will be written>
REPLICAS=<Number of TF serving replicas to deploy>
TF_SERVING_PORT=<Container port to use for TF serving>
USER_ID=<Your user ID>
GROUP_ID=<Your group ID>
```

Once you have edited the `mlops.env` file with your parameters,
deploy the training job using the command below. This command will
deploy resources to your default namespace. To use a different
namespace, specify `-n <namespace>` as part of your command.
```
kubectl -k wide-deep-large-ds-fp32-training/quickstart/k8s/mlops/pipeline apply
```

Depending on what version of kustomize is being used, you may get an
error reporting that a string was received instead of a integer. If this
is the case, the following command can be used to remove quotes that
are causing the issue:
```
kubectl kustomize wide-deep-large-ds-fp32-training/quickstart/k8s/mlops/pipeline | sed 's/runAsUser:.*"\([0-9]*\)"/runAsUser: \1/g' | sed 's/runAsGroup:.*"\([0-9]*\)"/runAsGroup: \1/g' | sed 's/fsGroup:.*"\([0-9]*\)"/fsGroup: \1/g' | sed 's/replicas:.*"\([0-9]*\)"/replicas: \1/g' | sed 's/containerPort:.*"\([0-9]*\)"/containerPort: \1/g' | kubectl apply -f -
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
usage: wide-deep-large-ds-fp32-training/quickstart/run_tf_serving_client.py [-h]
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
   python wide-deep-large-ds-fp32-training/quickstart/run_tf_serving_client.py -s <Internal IP>:<Node Port> -d <path to eval.csv> --b <batch size>
   ```
   The script will call the served model using data from the csv file
   and output performance and accuracy metrics.

##### Clean up the pipeline

To clean up the model training/serving pipeline, delete the service,
deployment, and other resources using the following commands:
```
kubectl -k wide-deep-large-ds-fp32-training/quickstart/k8s/mlops/pipeline delete
```

<!-- 61. Advanced Options -->

See the [Advanced Options for Model Packages and Containers](/quickstart/common/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

