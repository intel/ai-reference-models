<!--- 0. Title -->
# WaveNet FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running WaveNet FP32 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[wavenet-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/wavenet-fp32-inference.tar.gz)

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp32_inference.sh](/quickstart/text_to_speech/tensorflow/wavenet/inference/cpu/fp32/fp32_inference.sh) | Runs inference with a pretrained model |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* librosa==0.5
* numactl
* Clone the [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)
repo and get pull request #352 for the CPU optimizations.  The path to
the cloned repo will be passed as the model source directory when
running the launch script.

```
git clone https://github.com/ibab/tensorflow-wavenet.git
cd tensorflow-wavenet/

git fetch origin pull/352/head:cpu_optimized
git checkout cpu_optimized
```

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `TF_WAVENET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
export TF_WAVENET_DIR=<tensorflow-wavenet directory>
export OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/wavenet-fp32-inference.tar.gz
tar -xzf wavenet-fp32-inference.tar.gz
cd wavenet-fp32-inference

./quickstart/fp32_inference.sh
```

<!--- 60. Docker -->
## Docker

The model container intel/text-to-speech:tf-1.15.2-wavenet-fp32-inference includes the scripts and libraries needed to run 
WaveNet FP32 inference. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the output directory.

```
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/text-to-speech:tf-1.15.2-wavenet-fp32-inference \
  /bin/bash quickstart/fp32_inference.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

