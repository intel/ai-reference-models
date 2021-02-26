<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
<model name> <precision> <mode>.

To run the accuracy test, you will need
mount a volume and set the `DATASET_DIR` environment variable to point
to the prepped [ImageNet validation dataset](#dataset). The accuracy
script also downloads the pretrained model at runtime, so provide proxy
environment variables, if necessary.

```
DATASET_DIR=<path to the dataset folder>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --privileged --init -t \
  <docker image> \
  /bin/bash quickstart/fp32_accuracy.sh
```

Synthetic data is used when running batch or online inference, so no
dataset mount is needed.

```
docker run \
  --privileged --init -t \
  <docker image> \
  /bin/bash quickstart/<script name>.sh
```
