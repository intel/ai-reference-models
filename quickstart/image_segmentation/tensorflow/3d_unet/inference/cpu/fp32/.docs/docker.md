<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
<model name> <precision> <mode>. Prior to running the model in docker,
follow the [instructions above for downloading the BraTS dataset](#dataset).

1. Download the pretrained model from the
   [3DUnetCNN repo](https://github.com/ellisdg/3DUnetCNN/blob/ff5953b3a407ded73a00647f5c2029e9100e23b1/README.md#pre-trained-models).
   In this example, we are using the "Original U-Net" model, trained using the
   BraTS 2017 data.

1. To run one of the quickstart scripts using the model container, you'll need
   to provide volume mounts for the dataset, the directory where the pretrained
   model has been downloaded, and an output directory.

   ```
   DATASET_DIR=<path to the BraTS dataset>
   PRETRAINED_MODEL_DIR=<directory where the pretrained model has been downloaded>
   OUTPUT_DIR=<directory where log files will be written>
   # For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
   export BATCH_SIZE=<customized batch size value>

   docker run \
     --env DATASET_DIR=${DATASET_DIR} \
     --env OUTPUT_DIR=${OUTPUT_DIR} \
     --env BATCH_SIZE=${BATCH_SIZE} \
     --env PRETRAINED_MODEL=${PRETRAINED_MODEL_DIR}/tumor_segmentation_model.h5 \
     --env http_proxy=${http_proxy} \
     --env https_proxy=${https_proxy} \
     --volume ${DATASET_DIR}:${DATASET_DIR} \
     --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
     --volume ${PRETRAINED_MODEL_DIR}:${PRETRAINED_MODEL_DIR} \
     --privileged --init -t \
     <docker image> \
     /bin/bash quickstart/<script name>.sh
   ```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.
