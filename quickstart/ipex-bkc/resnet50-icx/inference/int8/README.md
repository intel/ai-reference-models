<!--- 0. Title -->
# IPEX ICX - ResNet50 INT8 

<!-- 10. Description -->
## Description

This container contains the ResNet50 model and is optimized for Intel's ICX architecture. The model runs on INT8 precision.
It contains the Intel Extension for Pytorch and PyTorch.

<!--- 20. Datasets -->
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

<!--- 30. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`inference_realtime.sh`](inference_realtime.sh) | Runs online inference using synthetic data (batch_size=1). |
| [`inference_throughput.sh`](inference_throughput.sh) | Runs batch inference using synthetic data (batch_size=512). |
| [`inference_accuracy.sh`](inference_accuracy.sh) | Measures the model accuracy (batch_size=128). |

<!--- 40. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
ResNet50 inference.

To run the accuracy test, you will need
mount a volume and set the `DATASET_DIR` environment variable to point
to the prepped [ImageNet validation dataset](#dataset). The accuracy
script also downloads the pretrained model at runtime, so provide proxy
environment variables, if necessary.

```
export DATASET_DIR=<path to the dataset folder>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env BASH_ENV=/root/.bash_profile \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --privileged --init -t \
  intel/image-recognition:pytorch-1.5.0-rc3-icx-a37fb5e8-resnet50-int8 \
  /bin/bash quickstart/inference_accuracy.sh
```

Note: When you run inference\_accuracy.sh and run into shared memory issue
You can use command described as below. Don't forget to change the \<shared memory value\> with
whatever value you want to keep for shared memory.

```
export DATASET_DIR=<path to the dataset folder>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env BASH_ENV=/root/.bash_profile \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --privileged --init -t \
  -shm-size <shared memory value> \
  intel/image-recognition:pytorch-1.5.0-rc3-icx-a37fb5e8-resnet50-int8 \
  /bin/bash quickstart/inference_accuracy.sh
```
To run throughput and realtime scripts you don't have to mount any dataset.
So, the command can be run as follows

```
docker run \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env BASH_ENV=/root/.bash_profile \
  --privileged --init -t \
  intel/image-recognition:pytorch-1.5.0-rc3-icx-a37fb5e8-resnet50-int8 \
  /bin/bash quickstart/<script name>
```

<!--- 50. License -->
## License

[LICENSE](/LICENSE)

