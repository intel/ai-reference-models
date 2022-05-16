<!--- 0. Title -->
# RFCN FP32 inference

<!-- 10. Description -->

This document has instructions for running RFCN FP32 inference using
Intel-optimized TensorFlow.


<!--- 20. Download link -->
## Download link

[rfcn-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/rfcn-fp32-inference.tar.gz)

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
| [`fp32_inference.sh`](/quickstart/object_detection/tensorflow/rfcn/inference/cpu/fp32/fp32_inference.sh) | Runs inference on a directory of raw images for 500 steps and outputs performance metrics. |
| [`fp32_accuracy.sh`](/quickstart/object_detection/tensorflow/rfcn/inference/cpu/fp32/fp32_accuracy.sh) | Processes the TF records to run inference and check accuracy on the results. |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* build-essential
* Cython
* contextlib2
* jupyter
* lxml
* matplotlib
* numpy>=1.17.4
* pillow>=8.1.2
* pycocotools

For more information, see the documentation on [prerequisites](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
in the TensorFlow models repo.

Download and untar the RFCN FP32 inference model package:

```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/rfcn-fp32-inference.tar.gz
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
[object detection installation instructions](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
from the [TensorFlow Model Garden](https://github.com/tensorflow/models) repository.

Once your environment is setup, navigate back to the directory that contains the RFCN FP32 inference
model package, set environment variables pointing to your dataset and output directories, and then run
a quickstart script.

To run inference with performance metrics:
```
DATASET_DIR=<path to the coco val2017 raw image directory (ex: /home/user/coco_dataset/val2017)>
OUTPUT_DIR=<directory where log files will be written>
TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>

./quickstart/fp32_inference.sh
```

To get accuracy metrics:
```
DATASET_DIR=<path to TF record file (ex: /home/user/coco_output/coco_val.record)>
OUTPUT_DIR=<directory where log files will be written>
TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>

./quickstart/fp32_accuracy.sh
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
DATASET_DIR=<path to the coco val2017 raw image directory (ex: /home/user/coco_dataset/val2017)>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/object-detection:tf-latest-rfcn-fp32-inference \
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
DATASET_DIR=<path to TF record file (ex: /home/user/coco_output/coco_val.record)>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/object-detection:tf-latest-rfcn-fp32-inference \
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

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

