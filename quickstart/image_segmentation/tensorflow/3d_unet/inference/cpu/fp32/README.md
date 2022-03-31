<!--- 0. Title -->
# 3D U-Net FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running 3D U-Net FP32 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[3d-unet-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/3d-unet-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

The following instructions are based on [BraTS2018 dataset preprocessing steps](https://github.com/ellisdg/3DUnetCNN/tree/update_to_brats18#tutorial-using-brats-data-and-python-3) in the [3D U-Net repository](https://github.com/ellisdg/3DUnetCNN/tree/update_to_brats18).
1. Download [BraTS2018 dataset](https://www.med.upenn.edu/sbia/brats2018/registration.html).
Please follow the steps to register and request the training and the validation data of the BraTS 2018 challenge.

2. Create a virtual environment and install the dependencies:
    ```
    # create a python3.6 based venv
    virtualenv --python=python3.6 brats18_env
    . brats18_env/bin/activate
    
    # install dependencies
    pip install intel-tensorflow==1.15.2
    pip install SimpleITK===1.2.0
    pip install keras==2.2.4
    pip install nilearn==0.6.2
    pip install tables==3.4.4
    pip install nibabel==2.3.3
    pip install nipype==1.7.0
    pip install numpy==1.16.3
    ```
    Install [ANTs N4BiasFieldCorrection](https://github.com/ANTsX/ANTs/releases/tag/v2.1.0) and add the location of the ANTs binaries to the PATH environmental variable:
    ```
    wget https://github.com/ANTsX/ANTs/releases/download/v2.1.0/Linux_Debian_jessie_x64.tar.bz2
    tar xvjf Linux_Debian_jessie_x64.tar.bz2
    cd debian_jessie
    export PATH=${PATH}:$(pwd)
    ```

3. Clone the [3D U-Net repository](https://github.com/ellisdg/3DUnetCNN/tree/update_to_brats18), and run the script for the dataset preprocessing:
    ```
    git clone https://github.com/ellisdg/3DUnetCNN.git
    cd 3DUnetCNN
    git checkout update_to_brats18
    
    # add the repository directory to the PYTHONPATH system variable
    export PYTHONPATH=${PWD}:$PYTHONPATH
    ```
    After downloading the dataset file `MICCAI_BraTS_2018_Data_Training.zip` (from step 1), place the unzipped folders in the `brats/data/original` directory.
    ```
    # extract the dataset
    mkdir -p brats/data/original && cd brats
    unzip MICCAI_BraTS_2018_Data_Training.zip -d data/original
    
    # import the conversion function and run the preprocessing:
    python
    >>> from preprocess import convert_brats_data
    >>> convert_brats_data("data/original", "data/preprocessed")
    
    # run training using the original UNet model to get `validation_ids.pkl` created in `brats` directory.
    python train.py 
    ```
After it finishes, set an environment variable to the path that contains the preprocessed dataset file `validation_ids.pkl`. 
```
export DATASET_DIR=/home/<user>/3DUnetCNN/brats
```

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
   wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/3d-unet-fp32-inference.tar.gz
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

