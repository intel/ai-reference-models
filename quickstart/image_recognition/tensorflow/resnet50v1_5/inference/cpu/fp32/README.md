<!--- 0. Title -->
# ResNet50 v1.5 FP32 inference

<!-- 10. Description -->

This document has instructions for running ResNet50 v1.5 FP32 inference using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to the TF records directory when running ResNet50 v1.5.

<!--- 20. Download link -->
## Download link

[resnet50v1-5-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50v1-5-fp32-inference.tar.gz)

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/fp32/fp32_online_inference.sh) | Runs online inference (batch_size=1). |
| [`fp32_batch_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/fp32/fp32_batch_inference.sh) | Runs batch inference (batch_size=128). |
| [`fp32_accuracy.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/fp32/fp32_accuracy.sh) | Measures the model accuracy (batch_size=100). |
| [`multi_instance_batch_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/fp32/multi_instance_batch_inference.sh) | Uses numactl to run batch inference (batch_size=128) with one instance per socket for 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |
| [`multi_instance_online_inference.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/fp32/multi_instance_online_inference.sh) | Uses numactl to run online inference (batch_size=1) using four cores per instance for 1500 steps and 50 warmup steps. If no `DATASET_DIR` is set, synthetic data is used. Waits for all instances to complete, then prints a summarized throughput value. |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package. The model package includes a
[pretrained model](https://zenodo.org/record/2535873/files/resnet50_v1.pb)
and the scripts needed to run the ResNet50 v1.5 FP32 <model>. Set
environment variables to point to the imagenet dataset directory (if real
data is being used), and an output directory where log files will be
written, then run a [quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50v1-5-fp32-inference.tar.gz
tar -xzf resnet50v1-5-fp32-inference.tar.gz
cd resnet50v1-5-fp32-inference

./quickstart/<script name>.sh
```


<!-- 60. Docker -->
## Docker

The model container `intel/image-recognition:tf-latest-resnet50v1-5-fp32-inference` includes the scripts
and libraries needed to run ResNet50 v1.5 FP32 inference. To run one of the model
inference quickstart scripts using this container, you'll need to provide volume mounts for
the ImageNet dataset (if a real dataset is being used) and an output directory where
checkpoint files will be written.

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
  intel/image-recognition:tf-latest-resnet50v1-5-fp32-inference \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!-- 61. Advanced Options -->
### Advanced Options

See the [Advanced Options for Model Packages and Containers](/quickstart/common/tensorflow/ModelPackagesAdvancedOptions.md)
document for more advanced use cases.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

