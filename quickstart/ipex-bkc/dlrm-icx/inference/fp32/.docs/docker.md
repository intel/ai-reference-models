<!--- 40. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
DLRM inference in fp32 precision.

To run the accuracy test, you will need
mount a volume and set the `DATASET_DIR` environment variable to point
to the prepped [Terabyte Click Logs Dataset](#dataset). 
The accuracy script needs the model weight. To set up the `WEIGHT_PATH` 
follow the steps described below.

```
export WEIGHT_PATH=<path to weight file>
wget -O $WEIGHT_PATH https://storage.googleapis.com/intel-optimized-tensorflow/models/icx-base-a37fb5e8/terabyte_mlperf_official.pt
```
Here, <path to weight file> can be the path you want the weight file to be downloaded at
Note that you'll need the `WEIGHT_PATH`only for calculating the accuracy

```
export DATASET_DIR=<path to the dataset folder>
export WEIGHT_PATH=<path to model weights>
docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env WEIGHT_PATH=${WEIGHT_PATH} \
  --env BASH_ENV=/root/.bash_profile \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${WEIGHT_PATH}:${WEIGHT_PATH} \
  --privileged --init -t \
  intel/recommendation:pytorch-1.5.0-rc3-icx-a37fb5e8-dlrm-fp32 \
  /bin/bash quickstart/inference_accuracy.sh
```

Model weight is not needed for rest of the scripts so you can use them as below,

```
export DATASET_DIR=<path to the dataset folder>
docker run \
  --env BASH_ENV=/root/.bash_profile \
  --env DATASET_DIR=${DATASET_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --privileged --init -t \
  intel/recommendation:pytorch-1.5.0-rc3-icx-a37fb5e8-dlrm-fp32 \
  /bin/bash quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.
