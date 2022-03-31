<!--- 0. Title -->
# ResNet50 BFloat16 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 BFloat16 inference using
[intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch).

<!--- 20. Download link -->
## Download link

[pytorch-resnet50-bfloat16-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/pytorch-resnet50-bfloat16-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

The [ImageNet](http://www.image-net.org/) validation dataset is used when
testing accuracy. The inference scripts use synthetic data, so no dataset
is needed.

Download and extract the ImageNet2012 dataset from http://www.image-net.org/,
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

The accuracy script looks for a folder named `val`, so after running the
data prep script, your folder structure should look something like this:

```
imagenet
└── val
    ├── ILSVRC2012_img_val.tar
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   ├── ILSVRC2012_val_00006697.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` directory should be set as the
`DATASET_DIR` when running accuracy
(for example: `export DATASET_DIR=/home/<user>/imagenet`).

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`bf16_online_inference.sh`](bf16_online_inference.sh) | Runs online inference using synthetic data (batch_size=1). |
| [`bf16_batch_inference.sh`](bf16_batch_inference.sh) | Runs batch inference using synthetic data (batch_size=128). |
| [`bf16_accuracy.sh`](bf16_accuracy.sh) | Measures the model accuracy (batch_size=128). |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch)
* [torchvision==v0.6.1](https://github.com/pytorch/vision/tree/v0.6.1)
* numactl

Download and untar the model package and then run a [quickstart script](#quick-start-scripts).

```
# Optional: to run accuracy script
export DATASET_DIR=<path to the preprocessed imagenet dataset>

# Download and extract the model package
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/pytorch-resnet50-bfloat16-inference.tar.gz
tar -xzf pytorch-resnet50-bfloat16-inference.tar.gz
cd pytorch-resnet50-bfloat16-inference

./quickstart/<script name>.sh
```

<!--- 60. Docker -->
## Docker

Use the base [PyTorch 1.8 container](https://hub.docker.com/layers/intel/intel-optimized-pytorch/1.8.0/images/sha256-5ca5d619b33bc6abc42cef654e9ee119ed0959c65f37de22a0bd8764c71412dd?context=explore)
`intel/intel-optimized-pytorch:1.8.0` to run ResNet50 BFloat16 inference.
To run the model quickstart scripts using the base PyTorch 1.8 container,
you will need to provide a volume mount for the pytorch-resnet50-bfloat16-inference package.

To run the accuracy test, you will need
mount a volume and set the `DATASET_DIR` environment variable to point
to the [ImageNet validation dataset](#dataset). The accuracy
script also downloads the pretrained model at runtime, so provide proxy
environment variables, if necessary.

```
DATASET_DIR=<path to the dataset folder>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume <path to the model package directory>:/pytorch-resnet50-bfloat16-inference \
  --privileged --init -it \
  intel/intel-optimized-pytorch:1.8.0 /bin/bash
```

Synthetic data is used when running batch or online inference, so no
dataset mount is needed.

```
docker run \
  --privileged --init -it \
  --volume <path to the model package directory>:/pytorch-resnet50-bfloat16-inference \
  intel/intel-optimized-pytorch:1.8.0 /bin/bash
```

Run quickstart scripts:
```
cd /pytorch-resnet50-bfloat16-inference
bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

