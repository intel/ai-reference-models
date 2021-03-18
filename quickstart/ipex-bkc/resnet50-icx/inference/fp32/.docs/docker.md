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
  intel/image-recognition:pytorch-1.5.0-rc3-icx-a37fb5e8-resnet50-fp32 \
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
  intel/image-recognition:pytorch-1.5.0-rc3-icx-a37fb5e8-resnet50-fp32 \
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
  intel/image-recognition:pytorch-1.5.0-rc3-icx-a37fb5e8-resnet50-fp32 \
  /bin/bash quickstart/<script name>
```
