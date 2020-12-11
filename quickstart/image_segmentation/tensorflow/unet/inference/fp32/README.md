<!--- 0. Title -->
# UNet FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running UNet FP32 inference using
Intel Optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[unet-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/unet-fp32-inference.tar.gz)

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp32_inference.sh](fp32_inference.sh) | Runs inference with a batch size of 1 using a pretrained model |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* numactl
* numpy==1.16.1
* Pillow>=7.1.0
* matplotlib
* click
* Clone the [tf_unet](https://github.com/jakeret/tf_unet) repository,
   and then get [PR #276](https://github.com/jakeret/tf_unet/pull/276)
   to get cpu optimizations:

   ```
   $ git clone https://github.com/jakeret/tf_unet.git

   $ cd tf_unet/

   $ git fetch origin pull/276/head:cpu_optimized

   $ git checkout cpu_optimized
   ``` 

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `TF_UNET_DIR` and an `OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
TF_UNET_DIR=<tensorflow-wavenet directory>
OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/unet-fp32-inference.tar.gz
tar -xzf unet-fp32-inference.tar.gz
cd unet_trained

quickstart/fp32_inference.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
UNet FP32 inference. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory.

```
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/image-segmentation:tf-1.15.2-imz-2.2.0-unet-fp32-inference \
  /bin/bash quickstart/fp32_inference.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

