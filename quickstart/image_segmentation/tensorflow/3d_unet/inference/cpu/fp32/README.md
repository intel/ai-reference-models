<!--- 0. Title -->
# 3D U-Net FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running 3D U-Net FP32 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[3d-unet-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_5_0/3d-unet-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

Follow the instructions at the [3D U-Net repository](https://github.com/ellisdg/3DUnetCNN)
for [downloading and preprocessing the BraTS dataset](https://github.com/ellisdg/3DUnetCNN/blob/ff5953b3a407ded73a00647f5c2029e9100e23b1/README.md#tutorial-using-brats-data-and-python-3).
The directory that contains the preprocessed dataset files will be passed to
the launch script when running the benchmarking script.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp32_inference.sh](fp32_inference.sh) | Runs inference with a batch size of 1 using the BraTS dataset and a pretrained model |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* numactl
* Keras==2.6.0rc3
* numpy==1.16.3
* nilearn==0.6.2
* tables==3.4.4
* nibabel==2.3.3
* SimpleITK===1.2.0
* h5py==2.10.0

Follow the [instructions above for downloading the BraTS dataset](#dataset).

1. Download the pretrained model from the
   [3DUnetCNN repo](https://github.com/ellisdg/3DUnetCNN/blob/ff5953b3a407ded73a00647f5c2029e9100e23b1/README.md#pre-trained-models).
   In this example, we are using the "Original U-Net" model, trained using the
   BraTS 2017 data.

2. Download and untar the model package. Set environment variables for the path
   to your `DATASET_DIR`, `PRETRAINED_MODEL` and `OUTPUT_DIR` (where log files
   will be written), and then run the [quickstart script](#quick-start-scripts).

   ```
   wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_5_0/3d-unet-fp32-inference.tar.gz
   tar -xzf 3d-unet-fp32-inference.tar.gz
   cd 3d-unet-fp32-inference

   export DATASET_DIR=<path to the BraTS dataset>
   export PRETRAINED_MODEL=<Path to the downloaded tumor_segmentation_model.h5 file>
   export OUTPUT_DIR=<directory where log files will be written>

   ./quickstart/fp32_inference.sh
   ```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
3D U-Net FP32 inference. Prior to running the model in docker,
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

   docker run \
     --env DATASET_DIR=${DATASET_DIR} \
     --env OUTPUT_DIR=${OUTPUT_DIR} \
     --env PRETRAINED_MODEL=${PRETRAINED_MODEL_DIR}/tumor_segmentation_model.h5 \
     --env http_proxy=${http_proxy} \
     --env https_proxy=${https_proxy} \
     --volume ${DATASET_DIR}:${DATASET_DIR} \
     --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
     --volume ${PRETRAINED_MODEL_DIR}:${PRETRAINED_MODEL_DIR} \
     --privileged --init -t \
     intel/image-segmentation:tf-1.15.2-3d-unet-fp32-inference \
     /bin/bash quickstart/<script name>.sh
   ```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

