<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the model training package to get the yaml and config
files for running inference on a single node using Kubernetes.
```
wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/wide-and-deep-large-ds-fp32-training.tar.gz
tar -xvf wide-and-deep-large-ds-fp32-training.tar.gz
```

### Execution

The model package includes a deployment that does 'mlops' (machine learning
operations) on kubernetes.
The directory tree within the model package is shown below:
```
quickstart/
├── common
│   └── k8s
│       └── mlops
│           ├── base
│           └── single-node
└── k8s
    └── mlops
        └── single-node
```

The deployments uses [kustomize](https://kustomize.io/) to configure
parameters. The parameters can be customized by editing kustomize
related files prior to deploying the single node inference job, which is
described in the [next section](#single-node-inference).

#### Single-node Training

Training is run by submitting a pod yaml file to the k8s api-server,
which results in the pod creation and then the specified
[quickstart script](#quick-start-scripts) is run in the pod's container.

Prior to running the job, edit the kustomize varaibles in the mlops.env
file. The mlops.env file for single node jobs is located at:
`wide-and-deep-large-ds-fp32-training/quickstart/k8s/mlops/single-node/mlops.env`.
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
kubectl -k wide-and-deep-large-ds-fp32-training/quickstart/k8s/mlops/single-node apply
```

Depending on what version of kustomize is being used, you may get an
error reporting that a string was received instead of a integer. If this
is the case, the following command can be used to remove quotes that
are causing the issue:
```
kubectl kustomize wide-and-deep-large-ds-fp32-training/quickstart/k8s/mlops/single-node | sed 's/runAsUser:.*"\([0-9]*\)"/runAsUser: \1/g' | sed 's/runAsGroup:.*"\([0-9]*\)"/runAsGroup: \1/g' | sed 's/fsGroup:.*"\([0-9]*\)"/fsGroup: \1/g' | kubectl apply -f -
```

Once the kubernetes job has been submitted, the pod status can be
checked using `kubectl get pods` and the logs can be viewed using
`kubectl logs -f wide-deep-large-ds-fp32-training`.

The script will write a log file, checkpoints, and the saved model to
the `OUTPUT_DIR`.

#### Cleanup

Clean up the model training job (delete the pod and other resources) using the following command:
```
kubectl -k wide-and-deep-large-ds-fp32-training/quickstart/k8s/mlops/single-node delete
```

### Advanced Options

See the [Advanced Options for Model Packages and Containers](ModelPackagesAdvancedOptions.md)
document for more advanced use cases.
