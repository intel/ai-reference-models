<!--- 0. Title -->
# SSD-MobileNet inference

<!-- 10. Description -->
## Description

This document has instructions for running SSD-MobileNet inference using
Intel-optimized TensorFlow.
<!--- 30. Datasets -->
## Datasets

The [COCO validation dataset](http://cocodataset.org) is used in these
SSD-Mobilenet quickstart scripts. The accuracy quickstart script require the dataset to be converted into the TF records format.
See the [COCO dataset](https://github.com/IntelAI/models/tree/master/datasets/coco) for instructions on
downloading and preprocessing the COCO validation dataset.

Set the `DATASET_DIR` to point to the dataset directory that contains the TF records file `coco_val.record` when running SSD-MobileNet accuracy script.
<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`inference.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference.sh) | Runs inference and outputs performance metrics. Uses synthetic data if no `DATASET_DIR` is set. Supported versions are (fp32, int8, bfloat16) |
| [`accuracy.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/accuracy.sh) | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required). Supported versions are (fp32, int8, bfloat16, bfloat32). |
| [`inference_throughput_multi_instance.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference_throughput_multi_instance.sh) | A multi-instance run that uses all the cores for each socket for each instance with a batch size of 448 and synthetic data. Supported versions are (fp32, int8, bfloat16, bfloat32) |
| [`inference_realtime_multi_instance.sh`](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/inference_realtime_multi_instance.sh) | A multi-instance run that uses 4 cores per instance with a batch size of 1. Uses synthetic data if no `DATASET_DIR` is set. Supported versions are (fp32, int8, bfloat16, bfloat32) |


<!-- 60. Docker -->
## Docker

When running in docker, the SSD-MobileNet inference container includes the
libraries and the model package, which are needed to run SSD-MobileNet inference. To run the quickstart scripts, you'll need to provide volume mounts for the
[COCO validation dataset](/datasets/coco/README.md) TF Record file and an output directory
where log files will be written. Omit the `DATASET_DIR` when running the multi-instance
quickstart scripts, since synthetic data will be used.

```
DATASET_DIR=<path to the dataset (for accuracy testing only)>
PRECISION=<set the precision to "int8" or "fp32">
OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRECISION=${PRECISION}
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/object-detection:tf-latest-ssd-mobilenet-inference \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)


