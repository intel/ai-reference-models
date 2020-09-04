<!--- 0. Title -->
# RFCN FP32 inference

<!-- 10. Description -->

This document has instructions for running RFCN FP32 inference using
Intel-optimized TensorFlow.


<!--- 20. Download link -->
## Download link

[rfcn-fp32-inference.tar.gz](https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/rfcn-fp32-inference.tar.gz)

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
| [`fp32_inference.sh`](fp32_inference.sh) | Runs inference on a directory of raw images for 500 steps and outputs performance metrics. |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | Processes the TF records to run inference and check accuracy on the results. |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
* [Kubernetes](#kubernetes)


<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, [prerequisites](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
to run the RFCN scripts must be installed in your environment.

Download and untar the RFCN FP32 inference model package:

```
wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/rfcn-fp32-inference.tar.gz
tar -xvf rfcn-fp32-inference.tar.gz
```

In addition to the general model zoo requirements, RFCN uses the object detection code from the
[TensorFlow Model Garden](https://github.com/tensorflow/models). Clone this repo with the SHA specified
below and apply the patch from the RFCN FP32 inference model package to run with TF2.

```
git clone https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b
git apply ../rfcn-fp32-inference/models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch
```

You must also install the dependencies and run the protobuf compilation described in the
[object detection installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#installation)
from the [TensorFlow Model Garden](https://github.com/tensorflow/models) repository.

Once your environment is setup, navigate back to the directory that contains the RFCN FP32 inference
model package, set environment variables pointing to your dataset and output directories, and then run
a quickstart script.

To run inference with performance metrics:
```
DATASET_DIR=<path to the coco val2017 directory>
OUTPUT_DIR=<directory where log files will be written>

quickstart/fp32_inference.sh
```

To get accuracy metrics:
```
DATASET_DIR=<path to the COCO validation TF record directory>
OUTPUT_DIR=<directory where log files will be written>

quickstart/fp32_accuracy.sh
```

<!-- 60. Docker -->
## Docker

When running in docker, the RFCN FP32 inference container includes the
libraries and the model package, which are needed to run RFCN FP32
inference. To run the quickstart scripts, you'll need to provide volume mounts for the
[COCO validation dataset](/dataset/coco/README.md) and an output directory
where log files will be written.

To run inference with performance metrics:
```
DATASET_DIR=<path to the coco val2017 directory>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-object-detection-rfcn-fp32-inference \
  /bin/bash quickstart/fp32_inference.sh
```

When the run completes, the log tail will note the average duration per step:

```
Avg. Duration per Step: ...
Ran inference with batch size 1
Log file location: ${OUTPUT_DIR}/benchmark_rfcn_inference_fp32_20200620_002239.log
```

To get accuracy metrics:
```
DATASET_DIR=<path to the COCO validation TF record directory>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-object-detection-rfcn-fp32-inference \
  /bin/bash quickstart/fp32_accuracy.sh
```

Below is a sample log file tail when running for accuracy:

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=10.41s).
Accumulating evaluation results...
DONE (t=1.62s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.532
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Ran inference with batch size 1
Log file location: ${OUTPUT_DIR}/benchmark_rfcn_inference_fp32_20200620_002841.log
```


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

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

