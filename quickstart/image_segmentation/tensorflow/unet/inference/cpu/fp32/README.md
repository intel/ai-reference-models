<!--- 0. Title -->
# UNet FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running UNet FP32 inference using
Intel Optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[unet-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/unet-fp32-inference.tar.gz)

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp32_inference.sh](fp32_inference.sh) | Runs inference with a batch size of 1 using a pretrained model |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/1.15.2/)
* numactl
* numpy==1.16.3
* Pillow>=8.1.2
* matplotlib
* click
* Clone the [tf_unet](https://github.com/jakeret/tf_unet) repository,
   and then get [PR #276](https://github.com/jakeret/tf_unet/pull/276)
   to get cpu optimizations:

   ```
   git clone https://github.com/jakeret/tf_unet.git
   cd tf_unet/
   git fetch origin pull/276/head:cpu_optimized
   git checkout cpu_optimized
   ``` 

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `TF_UNET_DIR` and an `OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
TF_UNET_DIR=<path to tf_unet directory>
OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/unet-fp32-inference.tar.gz
tar -xzf unet-fp32-inference.tar.gz
cd unet_trained

./quickstart/fp32_inference.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
UNet FP32 inference. To run one of the quickstart scripts 
using this container, you'll need to provide a volume mount for the 
output directory.

```
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/image-segmentation:tf-1.15.2-unet-fp32-inference \
  /bin/bash quickstart/fp32_inference.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

