<!--- 70. Kubernetes -->
## Kubernetes

Download and untar the model training package to get the yaml and config
files for running inference on a single node using Kubernetes.
```
wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/rfcn-fp32-inference.tar.gz
tar -xvf rfcn-fp32-inference.tar.gz
```

### Execution

The model package includes a deployment that does 'mlops' (machine learning
operations) on kubernetes.
The directory tree within the model package is shown below:
```
quickstart
├── common
│   └── tensorflow
│       └── k8s
│           └── mlops
│               ├── base
│               └── single-node
└── k8s
    └── mlops
        ├── pipeline
        └── single-node
```

The `pipeline` job can be used to preprocess the coco dataset to get a
TF records file and then run an RFCN FP32 accuracy test using an
[argo workflow](https://github.com/argoproj/argo). The `single-node`
uses a single pod to run inference to get performance metrics (using raw
images from the coco dataset) or test accuracy (when you already have
the TF records file on NFS).

The deployments use [kustomize](https://kustomize.io/) to configure
parameters. The parameters can be customized by editing kustomize
related files prior to deploying the job to kubernetes.

#### Single-node Inference

Inference is run by submitting a pod yaml file to the k8s api-server,
which results in the pod creation and then the specified
[quickstart script](#quick-start-scripts) is run in the pod's container.

Prior to running the job, edit the kustomize varaibles in the mlops.env
file. The mlops.env file for single node jobs is located at:
`rfcn-fp32-inference/quickstart/k8s/mlops/single-node/mlops.env`.
Key parameters to edit are:
```
DATASET_DIR=<path to the dataset directory>
MODEL_SCRIPT=<fp32_accuracy.sh or fp32_inference.sh>
NFS_MOUNT_PATH=<NFS mount path>
NFS_PATH=<NFS path>
NFS_SERVER=<IP address for your NFS Server>
OUTPUT_DIR=<Directory where log files will be written>
USER_ID=<Your user ID>
USER_NAME=<Your username>
GROUP_ID=<Your group ID>
GROUP_NAME=<Your group name>
```

> Note that when running inference, the `DATASET_DIR` should point to the
> directory of raw coco images (val2017) and when running accuracy testing,
> the `DATASET_DIR` should point to the TF records directory.

Once you have edited the `mlops.env` file with your parameters,
deploy the inference job using the following command:
```
kubectl -k rfcn-fp32-inference/quickstart/k8s/mlops/single-node apply
```

Depending on what version of kustomize is being used, you may get an
error reporting that a string was received instead of a integer. If this
is the case, the following command can be used to remove quotes that
are causing the issue:
```
kubectl kustomize rfcn-fp32-inference/quickstart/k8s/mlops/single-node | sed 's/runAsUser:.*"\([0-9]*\)"/runAsUser: \1/g' | sed 's/runAsGroup:.*"\([0-9]*\)"/runAsGroup: \1/g' | sed 's/fsGroup:.*"\([0-9]*\)"/fsGroup: \1/g' | kubectl apply -f -
```

Once the kubernetes job has been submitted, the pod status can be
checked using `kubectl get pods` and the logs can be viewed using
`kubectl logs -f <pod name>`.

##### Cleanup

Remove the workflow using the following command:
```
kubectl -k rfcn-fp32-inference/quickstart/k8s/mlops/single-node delete
```

#### Pipeline

The pipeline job uses an [Argo workflow](https://github.com/argoproj/argo)
to first convert the raw coco images to the TF records format and then
runs RFCN FP32 inference with an accuracy test using the TF records file.

The [COCO validation 2017 dataset and annotations](https://cocodataset.org/#download)
need to be downloaded to a directory on nfs. These will be used to create
the TF records file.

Prior to running the workflow, edit the kustomize varaibles in the mlops.env
file. The mlops.env file for workflow is located at:
`rfcn-fp32-inference/quickstart/k8s/mlops/pipeline/mlops.env`.
Key parameters to edit are:
```
WORKFLOW_NAME=<name for the workflow being deployed>
DATASET_DIR=<path to directory where the raw val2017 images and annotations are located>
OUTPUT_DIR=<Directory where log files will be written>
USER_ID=<Your user ID>
GROUP_ID=<Your group ID>
```

Once you have edited the `mlops.env` file with your parameters,
deploy the workflow using the following command:
```
kubectl -k rfcn-fp32-inference/quickstart/k8s/mlops/workflow apply
```

Depending on what version of kustomize is being used, you may get an
error reporting that a string was received instead of a integer. If this
is the case, the following command can be used to remove quotes that
are causing the issue:
```
kubectl kustomize rfcn-fp32-inference/quickstart/k8s/mlops/workflow | sed 's/runAsUser:.*"\([0-9]*\)"/runAsUser: \1/g' | sed 's/runAsGroup:.*"\([0-9]*\)"/runAsGroup: \1/g' | sed 's/fsGroup:.*"\([0-9]*\)"/fsGroup: \1/g' | kubectl apply -f -
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

##### Cleanup

Remove the workflow using the following command:
```
kubectl -k rfcn-fp32-inference/quickstart/k8s/mlops/pipeline delete
```

### Advanced Options

See the [Advanced Options for Model Packages and Containers](/quickstart/common/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.
