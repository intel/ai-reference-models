<!--- 0. Title -->
# SSD-MobileNet FP32 inference

<!-- 10. Description -->

This document has instructions for running SSD-MobileNet FP32 inference using
Intel-optimized TensorFlow.


<!--- 20. Download link -->
## Download link

[ssd-mobilenet-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/ssd-mobilenet-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Dataset

The [COCO validation dataset](http://cocodataset.org) is used in these
SSD-Mobilenet quickstart scripts. The inference and accuracy quickstart scripts require the dataset to be converted into the TF records format.
See the [COCO dataset](/datasets/coco/README.md) for instructions on
downloading and preprocessing the COCO validation dataset.


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_inference.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/fp32/fp32_inference.sh) | Runs inference on TF records and outputs performance metrics. |
| [`fp32_accuracy.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/fp32/fp32_accuracy.sh) | Processes the TF records to run inference and check accuracy on the results. |
| [`multi_instance_batch_inference.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/fp32/multi_instance_batch_inference.sh) | A multi-instance run that uses all the cores for each socket for each instance with a batch size of 448 and synthetic data. |
| [`multi_instance_online_inference.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/fp32/multi_instance_online_inference.sh) | A multi-instance run that uses 4 cores per instance with a batch size of 1. Uses synthetic data if no `DATASET_DIR` is set. |

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

To run inference, set environment variables with the path to the dataset
and an output directory, download and untar the SSD-MobileNet FP32
inference model package, and then run a [quickstart script](#quick-start-scripts).
```
DATASET_DIR=<path to the coco tf record file>
OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/ssd-mobilenet-fp32-inference.tar.gz
tar -xzf ssd-mobilenet-fp32-inference.tar.gz
cd ssd-mobilenet-fp32-inference

./quickstart/<script name>.sh
```

<!-- 60. Docker -->
## Docker

When running in docker, the SSD-MobileNet FP32 inference container includes the
libraries and the model package, which are needed to run SSD-MobileNet FP32
inference. To run the quickstart scripts, you'll need to provide volume mounts for the
[COCO validation dataset](/datasets/coco/README.md) TF Record file and an output directory
where log files will be written. Omit the `DATASET_DIR` when running the multi-instance
quickstart scripts, since synthetic data will be used.

```
DATASET_DIR=<path to the dataset (for accuracy testing only)>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/object-detection:tf-latest-ssd-mobilenet-fp32-inference \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)


