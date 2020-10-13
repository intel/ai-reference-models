<!--- 60. Docker -->
## Docker

The model container includes the pretrained model, scripts and libraries
needed to run  <model name> <precision> <mode>. To run one of the
quickstart scripts using this container, you'll need to provide a volume
mount for an output directory where logs will be written. If you are
testing accuracy, then the directory where the coco dataset
`validation-00000-of-00001` file located will also need to be mounted.

To run inference using synthetic data:
```
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  <docker image> \
  /bin/bash quickstart/int8_inference.sh
```

To test accuracy using the COCO dataset:
```
DATASET_DIR=<path to the COCO directory>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  <docker image> \
  /bin/bash quickstart/int8_accuracy.sh
```
